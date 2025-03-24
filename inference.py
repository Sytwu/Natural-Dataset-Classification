import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import Image_Dataset, valid_transform, load_dataset
from models import ResNeSt200

# Set CUDA device to GPU i
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def validate(models, valid_loader, device, weights):
    """
    Validate multiple models on a validation dataset and print
    both individual and ensemble accuracies.

    Parameters:
        models (list): List of trained model instances.
        valid_loader (DataLoader): Validation data loader.
        device (torch.device): Device for inference.
        weights (list): List of weights for each model in ensemble.

    Returns:
        None
    """
    # Prepare a list to store predictions from each model
    model_preds_list = [[] for _ in models]
    all_labels = []

    # Iterate over the validation loader
    for data in tqdm(valid_loader, desc="Validation"):
        inputs, labels = data
        inputs = inputs.to(device)

        # Accumulate ground truth labels
        all_labels.extend(labels.cpu().numpy())

        # Get predictions from each model
        for idx, model in enumerate(models):
            model.eval()
            with torch.no_grad():
                outputs = model(inputs)
            preds = F.softmax(outputs, dim=1).cpu().numpy()
            model_preds_list[idx].append(preds)

    all_labels = np.array(all_labels)

    # Calculate and print accuracy for each model
    for idx, preds in enumerate(model_preds_list):
        preds = np.concatenate(preds, axis=0)
        pred_labels = np.argmax(preds, axis=1)
        acc = np.mean(pred_labels == all_labels)
        print(f"Model {idx+1} Validation Acc: {acc:.4f}")

    # Calculate ensemble predictions using weighted sum
    ensemble_outputs = np.zeros_like(
        np.concatenate(model_preds_list[0], axis=0)
    )
    for w, preds in zip(weights, model_preds_list):
        preds = np.concatenate(preds, axis=0)
        ensemble_outputs += w * preds

    ensemble_pred_labels = np.argmax(ensemble_outputs, axis=1)
    ensemble_acc = np.mean(ensemble_pred_labels == all_labels)
    print(f"Ensemble Validation Acc: {ensemble_acc:.4f}")


def predict(models, test_loader, device, weights):
    """
    Generate ensemble predictions on test data.

    Parameters:
        models (list): List of trained model instances.
        test_loader (DataLoader): Test data loader.
        device (torch.device): Device for inference.
        weights (list): List of weights for each model in ensemble.

    Returns:
        preds (np.ndarray): Predicted labels for test data.
    """
    # Set all models to evaluation mode
    for model in models:
        model.eval()

    # Prepare a list to store predictions from each model
    model_preds_list = [[] for _ in models]

    # Iterate over the test loader
    for data in tqdm(test_loader, desc="Predicting"):
        inputs, _ = data
        inputs = inputs.to(device)

        with torch.no_grad():
            for idx, model in enumerate(models):
                outputs = model(inputs)
                preds = F.softmax(outputs, dim=1).cpu().numpy()
                model_preds_list[idx].append(preds)

    # Concatenate predictions from all batches for each model
    ensemble_outputs = [
        np.concatenate(preds, axis=0) for preds in model_preds_list
    ]

    # Compute weighted sum of predictions for ensemble output
    weighted_sum = sum(
        w * output for w, output in zip(weights, ensemble_outputs)
    )
    preds = np.argmax(weighted_sum, axis=1)

    return preds


if __name__ == "__main__":
    # Set up device for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # Load validation data and create validation DataLoader
    valid_images, valid_labels = load_dataset(usage="val")
    valid_dataset = Image_Dataset(
        valid_images, valid_labels, transform=valid_transform()
    )
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

    # Load test data and create test DataLoader; dummy labels are used
    images = load_dataset(usage="test")
    labels = [0] * len(images)
    test_dataset = Image_Dataset(
        images, labels, transform=valid_transform()
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Define model checkpoint paths for 10 folds
    model_paths = [f"model_best_merged_fold{i}.pth" for i in range(1, 11)]
    # Instantiate one model per checkpoint
    models = [ResNeSt200() for _ in model_paths]
    # Use equal weight for each model in the ensemble
    weights = [1.0 for _ in model_paths]

    # Load each model's state dictionary and move to the device
    for model, path in zip(models, model_paths):
        model.load_state_dict(torch.load(path))
        model.to(device)

    # Validate the ensemble of models
    validate(models, valid_loader, device, weights)

    # Generate predictions on the test dataset
    preds = predict(models, test_loader, device, weights)
    file_names = [
        os.path.splitext(os.path.basename(file))[0] for file in images
    ]

    # Save predictions to a CSV file
    df = pd.DataFrame({
        "image_name": file_names,
        "pred_label": preds
    })
    df.to_csv("prediction.csv", index=False)

    # Compress the CSV file into a zip archive
    os.system("zip -r solution.zip prediction.csv")
