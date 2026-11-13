#############################################################
#   FILE NAME: train.py
#
#   DESCRIPTION: This is the training file in which
#   the training process is defined. It fine-tunes the
#   ResNet-18 from model.py on a train split, scores it on
#   a val split each epoch, and saves the best checkpoint
#   (weights + class list) for the ModelSolver to load.
#
#   Run from the repository root:
#       python -m Library.train --synthetic
#       python -m Library.train --data Image-Data
#
#   Authors: Group 2
#############################################################

#############################################################
#   Libraries
#############################################################
from dataclasses import asdict, dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

#############################################################
#   Files Import
#############################################################

from .dataset import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SyntheticTileDataset,
    TileDataset,
)
from .model import create_model, save_checkpoint

#############################################################
#   Global Constants
#############################################################

DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_NUM_WORKERS = 4
DEFAULT_SEED = 0
DEFAULT_CHECKPOINT = "checkpoints/model.pt"

SYNTHETIC_TRAIN_SAMPLES = 512
SYNTHETIC_VAL_SAMPLES = 128
SYNTHETIC_VAL_SEED_OFFSET = 10_000

DEVICE_CHOICES = ("auto", "cuda", "cpu")

#############################################################
#   CLASSES
#############################################################

@dataclass
class TrainConfig:
    """
    CLASS NAME: TrainConfig

    DESCRIPTION: Holds every setting for one training run.
    It is saved inside the checkpoint so each run records
    how it was trained. The synthetic_* sizes are only used
    when synthetic is True.

    INPUTS: data_dir, synthetic, epochs, batch_size, lr,
    weight_decay, device, num_workers, seed, out_path,
    pretrained, synthetic_train_samples,
    synthetic_val_samples

    OUTPUTS: None
    """
    data_dir: str = "Image-Data"
    synthetic: bool = False
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    device: str = "auto"
    num_workers: int = DEFAULT_NUM_WORKERS
    seed: int = DEFAULT_SEED
    out_path: str = DEFAULT_CHECKPOINT
    pretrained: bool = True
    synthetic_train_samples: int = SYNTHETIC_TRAIN_SAMPLES
    synthetic_val_samples: int = SYNTHETIC_VAL_SAMPLES


@dataclass
class EpochStats:
    """
    CLASS NAME: EpochStats

    DESCRIPTION: The scores for one epoch. A list of these
    is the training history. val_accuracy is the fraction of
    (tile, class) decisions that are right at a 0.5
    threshold. seconds is ignored when comparing.

    INPUTS: epoch, train_loss, val_loss, val_accuracy, lr,
    seconds

    OUTPUTS: None
    """
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    lr: float
    seconds: float = field(default=0.0, compare=False)

#############################################################
#   FUNCTIONS
#############################################################

