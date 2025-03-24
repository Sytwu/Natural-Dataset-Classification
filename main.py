import os
from trainer import run

# Set the CUDA device to GPU i
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def default_config():
    """
    Return the default configuration dictionary for training.

    The configuration dictionary includes:
      - model: the model architecture name.
      - num_classes: number of output classes.
      - seed: random seed for reproducibility.
      - epochs: maximum number of training epochs.
      - batch_size: batch size for DataLoader.
      - base_lr: base learning rate for backbone parameters.
      - head_lr: learning rate for head layers.
      - weight_decay: weight decay factor for the optimizer.
      - patience: early stopping patience.
      - n_folds: number of folds for cross-validation.
      - fold_idx: index of the current validation fold.
      - is_cross_valid: flag to indicate cross-validation usage.
      - num_merge_weight: number of models for weight merging.
      - save_name: suffix for saved model files.

    Returns:
      config (dict): Default training configuration.
    """
    config = {
        "model": "ResNeSt200",
        "num_classes": 100,
        "seed": 111550159,
        "epochs": 40,
        "batch_size": 64,
        "base_lr": 1e-4,
        "head_lr": 1e-3,
        "weight_decay": 1e-5,
        "patience": 10,
        "n_folds": 10,
        "fold_idx": 0,
        "is_cross_valid": True,
        "num_merge_weight": 2,
        "save_name": "",
    }
    return config


if __name__ == "__main__":
    # Load the default configuration
    config = default_config()
    n_folds = 10
    accs = []

    # Set the number of merged models to 1 for this run
    config["num_merge_weight"] = 1

    # Iterate through each fold for cross-validation
    for fold_idx in range(n_folds):
        print(f"Start with Fold {fold_idx + 1}")
        config["fold_idx"] = fold_idx
        config["save_name"] = f"_new_fold{fold_idx + 1}"
        accs.append(run(config))
        print("-" * 100)
