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
    # 1. Check the head matches the class list:
    #    model.fc.out_features must equal len(classes),
    #    otherwise raise ValueError.

    # 2. If thresholds is given, check every key is in classes
    #    (raise ValueError on an unknown class).

    # 3. Copy the weights to CPU so the file loads on a machine
    #    without a GPU:
    #    {name: tensor.detach().cpu() for name, tensor in state_dict}

    # 4. Build the checkpoint dict:
    #    "arch"        -> MODEL_ARCH
    #    "classes"     -> list(classes)
    #    "thresholds"  -> thresholds (or None)
    #    "model_state" -> CPU state dict from step 3
    #    then merge in extra, but don't let it overwrite the
    #    four keys above (raise ValueError if it tries).

    # 5. Create the parent folder if needed (os.makedirs with
    #    exist_ok=True; skip when the path has no folder).

    # 6. torch.save(checkpoint, path)


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
    # 1. torch.load(path, map_location="cpu", weights_only=True)
    #    weights_only=True is why extra must hold only plain
    #    Python values (no dataclasses or custom objects).

    # 2. Check checkpoint["arch"] == MODEL_ARCH, otherwise
    #    raise ValueError.

    # 3. If expected_classes is given, compare it to
    #    checkpoint["classes"] in content AND order; raise
    #    ValueError showing both lists if they differ.

    # 4. model = create_model(num_classes=len(classes),
    #                         pretrained=False)

    # 5. model.load_state_dict(checkpoint["model_state"])

    # 6. Move to device and switch to eval mode.

    # 7. return model, checkpoint
