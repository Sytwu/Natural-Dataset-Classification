def get_optimizer_params(model, base_lr, head_lr):
    groups = {
        "layer1": [],
        "layer2": [],
        "layer3": [],
        "layer4": [],
        "head": [],
    }
    for name, param in model.named_parameters():
        if "layer1" in name:
            groups["layer1"].append(param)
        elif "layer2" in name:
            groups["layer2"].append(param)
        elif "layer3" in name:
            groups["layer3"].append(param)
        elif "layer4" in name:
            groups["layer4"].append(param)
        elif "resnest.fc" in name:
            groups["head"].append(param)

    return [
        {"params": groups["layer1"], "lr": base_lr},
        {"params": groups["layer2"], "lr": base_lr},
        {"params": groups["layer3"], "lr": base_lr},
        {"params": groups["layer4"], "lr": base_lr},
        {"params": groups["head"], "lr": head_lr},
    ]
