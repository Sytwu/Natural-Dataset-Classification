import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from dataloader import Image_Dataset, load_dataset
from dataloader import train_transform, valid_transform
from dataloader import only_cutmix
from model_stock import merge_multi
from models import get_model
from optim import get_optimizer_params
from utils import set_seed


def train(model, train_loader, criterion, optimizer, scaler, device):
    """
    Train a single model for one epoch.

    Parameters:
        model: The neural network to train.
        train_loader: DataLoader providing training data.
        criterion: Loss function (e.g., cross entropy).
        optimizer: Optimizer for updating model weights.
        scaler: GradScaler for mixed precision training.
        device: The device (CPU or CUDA) where the model is located.

    Returns:
        Average training loss over the epoch.
    """
    model.train()  # Set the model to training mode
    running_loss = 0.0
    # Use tqdm for a progress bar over the training batches
    bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, data in bar:
        inputs, labels = data
        # Apply CutMix augmentation to the inputs and labels
        inputs, labels = only_cutmix()(inputs, labels)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Clear previous gradients

        # Enable autocasting for mixed precision training on CUDA
        with autocast("cuda"):
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()  # Accumulate loss
        bar.set_description(f"Training Loss: {loss.item():.5f}")

    return running_loss / len(train_loader)  # Return average loss


def validate(model, valid_loader, criterion, device):
    """
    Validate the model on a validation set.

    Parameters:
        model: The neural network to validate.
        valid_loader: DataLoader providing validation data.
        criterion: Loss function.
        device: The device (CPU or CUDA) where the model is located.

    Returns:
        Average validation loss and accuracy.
    """
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0
    bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    with torch.no_grad():  # Disable gradient computation for validation
        for i, data in bar:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast("cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Compute predictions and update accuracy statistics
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            bar.set_description(f"Loss: {loss.item():.5f}")

    avg_loss = running_loss / len(valid_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def run(config):
    """
    Main function to run training and validation.

    The configuration dictionary 'config' includes parameters such as:
        'model': model architecture name (e.g., 'ResNeSt200'),
        'num_classes': number of output classes,
        'seed': random seed for reproducibility,
        'epochs': maximum number of training epochs,
        'batch_size': batch size for DataLoader,
        'base_lr': base learning rate for backbone parameters,
        'head_lr': learning rate for the head layers,
        'weight_decay': weight decay for the optimizer,
        'patience': early stopping patience,
        'n_folds': number of folds for cross validation,
        'fold_idx': the index of the current validation fold,
        'is_cross_valid': flag indicating whether to use cross validation,
        'num_merge_weight': number of models for weight merging,
        'save_name': string suffix to add to saved model files.

    Returns:
        The best merged model accuracy achieved on the validation set.
    """
    save_name = config["save_name"]
    set_seed(config["seed"])

    # Load training and validation datasets
    full_train_images, full_train_labels = load_dataset(usage="train")
    valid_images, valid_labels = load_dataset(usage="val")

    # If cross validation is enabled, split the training set accordingly
    if config["is_cross_valid"]:
        num_samples = len(full_train_images)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        folds = np.array_split(indices, config["n_folds"])
        fold_valid_idx = folds[config["fold_idx"]]
        fold_train_idx = np.concatenate(
            [fold for i, fold in enumerate(folds)
             if i != config["fold_idx"]]
        )
        # Combine validation images/labels from original validation set and
        # current fold
        train_images_fold = (np.array(full_train_images)[fold_train_idx]
                             .tolist())
        train_labels_fold = (np.array(full_train_labels)[fold_train_idx]
                             .tolist())
        valid_images += (np.array(full_train_images)[fold_valid_idx]
                         .tolist())
        valid_labels += (np.array(full_train_labels)[fold_valid_idx]
                         .tolist())
    else:
        train_images_fold = full_train_images
        train_labels_fold = full_train_labels

    # Create dataset objects with appropriate transforms
    train_dataset = Image_Dataset(
        train_images_fold, train_labels_fold, transform=train_transform()
    )
    valid_dataset = Image_Dataset(
        valid_images, valid_labels, transform=valid_transform()
    )

    # Create DataLoader objects for training and validation
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Select device: GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Determine the number of models to train for weight merging
    num_models = config.get("num_merge_weight", 2)

    # Initialize multiple models, their optimizers, and gradient scalers for
    # mixed precision training
    models = [
        get_model(config["model"], config["num_classes"]).to(device)
        for _ in range(num_models)
    ]
    optimizers = [
        optim.AdamW(
            get_optimizer_params(model, config["base_lr"], config["head_lr"]),
            weight_decay=config["weight_decay"],
        )
        for model in models
    ]
    scalers = [GradScaler("cuda") for _ in range(num_models)]

    # Lists to track the best accuracy for individual models and the merged
    # model
    best_acc_list = [0.0] * num_models
    best_acc_merged = 0.0
    patience_counter = 0  # Counter for early stopping
    train_losses = []  # List to store average training loss per epoch
    valid_losses = []  # List to store average validation loss per epoch
    epochs = config["epochs"]

    # Start training loop over epochs
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_train_losses = []
        epoch_valid_losses = []
        epoch_valid_accs = []
        # Train and validate each model individually
        for i in range(num_models):
            # Adjust seed for each model training iteration to ensure varied
            # randomness
            seed_i = config["seed"] + epoch * num_models + i
            set_seed(seed_i)

            # Train the i-th model for one epoch
            loss = train(models[i], train_loader, criterion, optimizers[i],
                         scalers[i], device)
            # Validate the i-th model on the validation set
            val_loss, val_acc = validate(models[i], valid_loader, criterion,
                                         device)
            print(
                f"Model {i + 1}: Training Loss: {loss:.4f}, Validation Loss: "
                f"{val_loss:.4f}, Acc: {val_acc:.4f}"
            )

            epoch_train_losses.append(loss)
            epoch_valid_losses.append(val_loss)
            epoch_valid_accs.append(val_acc)

            # Save the model if it has the best accuracy so far for this model
            if val_acc > best_acc_list[i]:
                best_acc_list[i] = val_acc
                torch.save(
                    models[i].state_dict(), f"model_best_{i+1}{save_name}.pth"
                )
                print(f"Saved best Model {i+1} at epoch {epoch+1}")

        # Merge all models' weights using the provided merge_multi function
        model_states = [model.state_dict() for model in models]
        merged_state = merge_multi(model_states)

        # Create a new model instance for the merged weights and load the state
        merged_model = get_model(config["model"], config["num_classes"]).to(
            device
        )
        merged_model.load_state_dict(merged_state)
        # Validate the merged model
        val_loss_merged, val_acc_merged = validate(
            merged_model, valid_loader, criterion, device
        )
        print(
            f"Merged Model - Validation Loss: {val_loss_merged:.4f}, Acc: "
            f"{val_acc_merged:.4f}"
        )

        # Save merged model if it has the best validation accuracy so far;
        # otherwise, update the patience counter
        if val_acc_merged > best_acc_merged:
            best_acc_merged = val_acc_merged
            patience_counter = 0
            torch.save(merged_state, f"model_best_merged{save_name}.pth")
            print(f"Saved best Merged Model at epoch {epoch+1}")
        else:
            patience_counter += 1

        # Early stopping check: if no improvement for a number of epochs equal
        # to 'patience'
        if patience_counter >= config["patience"]:
            print("Early stopping")
            break

        # Update each individual model with the merged weights to synchronize
        # them
        for model in models:
            model.load_state_dict(merged_state)

        # Compute and record the average training and validation losses for the
        # epoch
        avg_train_loss = sum(epoch_train_losses) / num_models
        avg_valid_loss = sum(epoch_valid_losses) / num_models
        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)

    print("\nBest Merged Accuracy: {:.4f}".format(best_acc_merged))
    return best_acc_merged
