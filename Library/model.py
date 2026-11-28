#############################################################
#   FILE NAME: model.py
#
#   DESCRIPTION: Builds the ResNet-18 tile classifier, runs
#   multi-label predictions on a batch of tiles, and saves /
#   loads model checkpoints (weights + class list +
#   thresholds) so ModelSolver can use a trained model
#   without importing any training code.
#
#   Authors: Group 2
#############################################################

#############################################################
#   Libraries
#############################################################
import os
import torch
import torch.nn as nn
from torchvision import models

#############################################################
#   Global Constants
#############################################################
MODEL_ARCH = "resnet18"

#############################################################
#   FUNCTIONS
#############################################################
def create_model(num_classes=8, pretrained=True):
    """
    FUNCTION NAME: create_model

    DESCRIPTION: Loads torchvision ResNet-18 (ImageNet weights
    when pretrained) and replaces the final fc layer with an
    N-way linear head for multi-label classification. Always
    pass num_classes=len(dataset.classes).

    INPUTS: num_classes (int), pretrained (bool)

    OUTPUTS: model (torch.nn.Module)
    """
    # Load ResNet-18 with pretrained ImageNet weights
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Replace the original classification layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def predict(model, images, device="cpu"):
    """
    FUNCTION NAME: predict

    DESCRIPTION: Moves the model to device, switches it to eval
    mode, and returns one independent sigmoid probability per
    class for each image (multi-label, not softmax).

    INPUTS: model (torch.nn.Module), images (Nx3x224x224 float
    tensor, already transformed with dataset.get_transform()),
    device (str or torch.device)

    OUTPUTS: probabilities (NxC float tensor, values 0-1)
    """
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        images = images.to(device)
        logits = model(images)
        probabilities = torch.sigmoid(logits)

    return probabilities


def save_checkpoint(path, model, classes, thresholds=None, extra=None):
    """
    FUNCTION NAME: save_checkpoint

    DESCRIPTION: Writes a .pt checkpoint dict holding the
    architecture name, the ordered class list, per-class
    thresholds (None until tuned), the model weights on CPU,
    and any extra keys (config, history, best epoch, ...).
    Called by train.py every time val loss improves.

    INPUTS: path (str), model (torch.nn.Module), classes (list
    of str, in head order), thresholds (dict of class -> float,
    or None), extra (dict of plain Python values, or None)

    OUTPUTS: None
    """
    # 1. The head size must match the class list
    if model.fc.out_features != len(classes):
        raise ValueError(
            f"Model head has {model.fc.out_features} outputs but "
            f"{len(classes)} classes were given."
        )

    # 2. Every threshold must name a known class
    if thresholds is not None:
        unknown = [name for name in thresholds if name not in classes]
        if unknown:
            raise ValueError(f"Thresholds name unknown classes: {unknown}")
        thresholds = {name: float(value) for name, value in thresholds.items()}

    # 3. Copy the weights to CPU so the file loads without a GPU
    model_state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }

    # 4. Build the checkpoint dict; extra may not overwrite the core keys
    checkpoint = {
        "arch": MODEL_ARCH,
        "classes": list(classes),
        "thresholds": thresholds,
        "model_state": model_state,
    }
    if extra:
        clashes = [key for key in extra if key in checkpoint]
        if clashes:
            raise ValueError(f"extra may not overwrite reserved keys: {clashes}")
        checkpoint.update(extra)

    # 5. Create the parent folder if needed
    folder = os.path.dirname(str(path))
    if folder:
        os.makedirs(folder, exist_ok=True)

    # 6. Write it
    torch.save(checkpoint, path)


def load_checkpoint(path, expected_classes=None, device="cpu"):
    """
    FUNCTION NAME: load_checkpoint

    DESCRIPTION: Loads a checkpoint written by save_checkpoint,
    rebuilds the model with pretrained=False (no download),
    loads its weights, and returns it in eval mode on device.
    Optionally checks the class list matches what the caller
    (e.g. ModelSolver / the tile source) expects.

    INPUTS: path (str), expected_classes (list of str, or
    None), device (str or torch.device)

    OUTPUTS: (model, checkpoint) - the ready-to-use model and
    the full checkpoint dict (classes, thresholds, extras)
    """
    # 1. weights_only=True only allows plain Python values and tensors
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    # 2. Check the architecture
    if checkpoint.get("arch") != MODEL_ARCH:
        raise ValueError(
            f"Checkpoint arch is {checkpoint.get('arch')!r}, expected {MODEL_ARCH!r}."
        )

    # 3. Check the classes match in content and order
    classes = checkpoint["classes"]
    if expected_classes is not None and list(expected_classes) != list(classes):
        raise ValueError(
            f"Checkpoint classes {list(classes)} do not match expected "
            f"classes {list(expected_classes)}."
        )

    # 4-5. Rebuild the model without downloading weights, then load them
    model = create_model(num_classes=len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])

    # 6. Move to device and switch to eval mode
    model = model.to(device)
    model.eval()

    # 7. Return the model and the full checkpoint
    return model, checkpoint
