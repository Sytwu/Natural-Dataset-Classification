import torch
import numpy as np
from collections import OrderedDict

# Import custom model definitions
from models import ResNeXt101

EPS = 1e-8  # Small constant to avoid division by zero


# ============================
# 1. Weight Loading Function
# ============================
def load_state_dict(path):
    """
    Load a state dictionary from the given file path.

    Parameters:
      path (str): File path to the saved weights.

    Returns:
      state_dict (dict): Dictionary containing model parameters.
    """
    return torch.load(path, map_location="cpu")


# ============================================================
# 2. Model Stock Weight Fusion Functions
#    (a) compute_angle: Compute the angle between each layer's
#        fine-tuned weights and the pretrained weights (in degrees).
#    (b) compute_ratio: Calculate a fusion ratio for each layer based
#        on the angle using a formula with k=2.
#    (c) merge: Merge the fine-tuned weights (after averaging)
#        with the pretrained weights using the computed ratio.
# ============================================================
def compute_angle(state_dict_1, state_dict_2, ref_state_dict,
                  add_ignore_keys=None, return_cos=False):
    """
    Compute the angle (or cosine similarity) between the difference
    of two fine-tuned state dictionaries and a reference
    pretrained state dictionary.

    For each layer:
      - Compute difference vectors (fine-tuned minus pretrained).
      - Calculate the cosine similarity between these vectors.
      - Convert the cosine similarity to an angle in degrees
        (unless return_cos is True).

    Parameters:
      state_dict_1 (dict): First fine-tuned model's state dict.
      state_dict_2 (dict): Second fine-tuned model's state dict.
      ref_state_dict (dict): Reference state dict (pretrained weights).
      add_ignore_keys (list): Keys to ignore during computation.
      return_cos (bool): If True, return cosine similarity instead
                         of angle.

    Returns:
      return_dict (OrderedDict): Maps each key to its angle (in degrees)
      or cosine similarity.
    """
    if add_ignore_keys is None:
        add_ignore_keys = []
    ignore_keys = []
    ignore_keys += ["module." + key for key in add_ignore_keys]
    ignore_keys.extend(add_ignore_keys)

    return_dict = OrderedDict()
    with torch.no_grad():
        for key in ref_state_dict:
            if key in ignore_keys:
                continue
            if key not in state_dict_1 or key not in state_dict_2:
                continue
            state_dict_1_val = state_dict_1[key]
            state_dict_2_val = state_dict_2[key]
            ref_val = ref_state_dict[key]
            if not (state_dict_1_val.shape == state_dict_2_val.shape ==
                    ref_val.shape):
                continue

            vector1 = (state_dict_1_val - ref_val).float()
            vector2 = (state_dict_2_val - ref_val).float()
            cosine_val = (torch.sum(vector1 * vector2) /
                          (torch.sqrt(torch.sum(vector1 ** 2) *
                                      torch.sum(vector2 ** 2)) + EPS))
            cosine_val = torch.clamp(cosine_val, min=-1.0, max=1.0)
            if return_cos:
                return_dict[key] = cosine_val
            else:
                return_dict[key] = np.rad2deg(
                    torch.acos(cosine_val).detach().cpu())
    return return_dict


def compute_ratio(angle_dict, k=2):
    """
    Compute the fusion ratio for each layer based on its angle.

    The formula is:
      ratio = k * cos(angle) / [ (k-1) * cos(angle) + 1 ]
    where the angle is converted to radians.

    Parameters:
      angle_dict (dict): Maps each layer key to its angle (in degrees).
      k (float): Hyperparameter for fusion ratio (default is 2).

    Returns:
      ratio_dict (dict): Maps each layer key to its fusion ratio.
    """
    ratio_dict = {}
    for key, angle_deg in angle_dict.items():
        angle_rad = np.deg2rad(angle_deg)
        ratio_dict[key] = (k * np.cos(angle_rad) /
                           (((k - 1) * np.cos(angle_rad)) + 1 + EPS))
    return ratio_dict


def merge(w_ft1, w_ft2, w_pt, ratio):
    """
    Merge fine-tuned weights with pretrained weights using the
    fusion ratio.

    Steps:
      1. Average the two fine-tuned weights to obtain w12.
      2. For each layer:
           w_merge = ratio * w12 + (1 - ratio) * w_pt

    Parameters:
      w_ft1 (dict): First fine-tuned model's state dict.
      w_ft2 (dict): Second fine-tuned model's state dict.
      w_pt (dict): Pretrained model's state dict.
      ratio (dict): Maps each layer key to its fusion ratio.

    Returns:
      w_merge (dict): The merged state dictionary.
    """
    w12 = {}
    for key in w_ft1.keys():
        w12[key] = (w_ft1[key].clone() + w_ft2[key].clone()) / 2.0

    w_merge = w12.copy()
    for key, r in ratio.items():
        if key in w_merge and key in w_pt:
            w_merge[key] = w12[key] * r + w_pt[key] * (1.0 - r)
    return w_merge


def merge_multi(w_fts):
    """
    Merge multiple fine-tuned state dictionaries by averaging.

    Parameters:
      w_fts (list): List of fine-tuned state dictionaries.

    Returns:
      w_center (dict): Averaged (merged) state dictionary.
    """
    w_center = {}
    for key in w_fts[0].keys():
        s = 0
        for w_ft in w_fts:
            s += w_ft[key].clone()
        w_center[key] = s / len(w_fts)
    return w_center


# =============================================================
# 3. Main Process: Load Weights, Compute Angles and Ratios,
#    Merge Weights, and Load into a ResNeXt101 Model.
# =============================================================
if __name__ == "__main__":
    # Define file paths for pretrained and fine-tuned weights.
    # Assume three files: one for pretrained and two for fine-tuned.
    pretrained_weight = "resnext101_pt.pth"
    finetuned_weight_1 = "resnext101_ft1.pth"
    finetuned_weight_2 = "resnext101_ft2.pth"

    # Load state dictionaries from the files.
    weight_pt = load_state_dict(pretrained_weight)
    weight_ft1 = load_state_dict(finetuned_weight_1)
    weight_ft2 = load_state_dict(finetuned_weight_2)

    # Compute the angle between fine-tuned and pretrained weights.
    angle = compute_angle(weight_ft1, weight_ft2, weight_pt)
    # Compute the fusion ratio for each layer (using k=2).
    ratio = compute_ratio(angle, k=2)
    print("Computed ratio keys:", list(ratio.keys()))

    # Merge weights: average fine-tuned weights then fuse with
    # pretrained weights using the ratio.
    merged_weight = merge(weight_ft1, weight_ft2, weight_pt, ratio)

    # Initialize a ResNeXt101 model with 100 classes and load weights.
    model = ResNeXt101(num_classes=100)
    model.load_state_dict(merged_weight)
    model.eval()  # Set model to evaluation mode

    # The merged model is now ready for inference or evaluation.
    print("ResNeXt101 Model Stock fusion completed!")
