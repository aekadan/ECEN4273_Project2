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
import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
import torch.nn as nn
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
    get_transform,
)
from .model import create_model, save_checkpoint, load_checkpoint

#############################################################
#   Global Constants
#############################################################

DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 128    # Reduce to 64 if you run out of GPU memory
DEFAULT_LR = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_NUM_WORKERS = 4
DEFAULT_SEED = 0
DEFAULT_CHECKPOINT = "checkpoints/model.pt"

SYNTHETIC_TRAIN_SAMPLES = 512
SYNTHETIC_VAL_SAMPLES = 128
SYNTHETIC_VAL_SEED_OFFSET = 10_000

DEVICE_CHOICES = ("auto", "cuda", "cpu")
PREDICTION_THRESHOLD = 0.5



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

def get_train_transform():
    """
    FUNCTION NAME: get_train_transform

    DESCRIPTION: Training augmentation for tiles.

    INPUTS: None

    OUTPUTS: torchvision transforms.Compose
    """
    RandomResizedCrop = transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0), ratio=(3/4, 4/3))
    RandomHorizontalFlip = transforms.RandomHorizontalFlip()
    ColorJitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    ToTensor = transforms.ToTensor()
    Normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    return transforms.Compose([RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize])
    

def select_device(requested):
    """
    FUNCTION NAME: select_device

    DESCRIPTION: Turns "auto", "cuda", or "cpu" into a torch.device.

    INPUTS: requested (str)

    OUTPUTS: torch.device
    """
    if requested not in DEVICE_CHOICES:
        raise ValueError(f"Invalid device choice: {requested}. Must be one of {DEVICE_CHOICES}.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but no GPU is available.")
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    """
    FUNCTION NAME: set_seed

    DESCRIPTION: Seeds Python and torch so runs are repeatable.

    INPUTS: seed (int)

    OUTPUTS: None
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_config(config : TrainConfig):
    """
    FUNCTION NAME: validate_config

    DESCRIPTION: Raises ValueError on bad settings before any
    work is done.

    INPUTS: config (TrainConfig)

    OUTPUTS: None
    """
    if not isinstance(config.epochs, int) or config.epochs < 1:
        raise ValueError("Epochs must be an integer >= 1.")
    if not isinstance(config.batch_size, int) or config.batch_size < 1:
        raise ValueError("Batch size must be an integer >= 1.")
    if config.lr <= 0:
        raise ValueError("Learning rate must be > 0.")
    if config.weight_decay < 0:
        raise ValueError("Weight decay must be >= 0.")
    if config.num_workers < 0:
        raise ValueError("Number of workers must be >= 0.")
    if config.synthetic_train_samples < 1:
        raise ValueError("Synthetic train samples must be >= 1.")
    if config.synthetic_val_samples < 1:
        raise ValueError("Synthetic val samples must be >= 1.")


def build_datasets(config : TrainConfig):
    """
    FUNCTION NAME: build_datasets

    DESCRIPTION: Builds the train and val datasets.

    INPUTS: config (TrainConfig)

    OUTPUTS: (train_dataset, val_dataset)
    """
    if config.synthetic:
        train_synthetic_dataset: SyntheticTileDataset = SyntheticTileDataset(num_samples=config.synthetic_train_samples,
                                                                             seed = config.seed, 
                                                                             transform=get_train_transform())
        val_synthetic_dataset : SyntheticTileDataset = SyntheticTileDataset(num_samples=config.synthetic_val_samples, 
                                                                            seed=config.seed + SYNTHETIC_VAL_SEED_OFFSET, 
                                                                            transform=get_transform())
        return train_synthetic_dataset, val_synthetic_dataset

    data_dir = Path(config.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.is_dir():
        raise FileNotFoundError(f"{train_dir} does not exist")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"{val_dir} does not exist")

    train_tile_dataset: TileDataset = TileDataset(data_dir=train_dir, transform=get_train_transform())
    val_tile_dataset: TileDataset = TileDataset(data_dir=val_dir, classes=train_tile_dataset.classes, transform=get_transform())

    return train_tile_dataset, val_tile_dataset


def build_loaders(config: TrainConfig, train_dataset, val_dataset, device):
    """
    FUNCTION NAME: build_loaders

    DESCRIPTION: Wraps the datasets in DataLoaders.

    INPUTS: config, train_dataset, val_dataset, device

    OUTPUTS: (train_loader, val_loader)
    """
    pin_memory = (device.type == "cuda")
    persistent_workers = config.num_workers > 0

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              generator=torch.Generator().manual_seed(config.seed),
                              num_workers=config.num_workers,
                              pin_memory=pin_memory,
                              persistent_workers=persistent_workers)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=config.num_workers,
                            pin_memory=pin_memory,
                            persistent_workers=persistent_workers)

    return train_loader, val_loader

def build_optimizer(model, config, steps_per_epoch):
    """
    FUNCTION NAME: build_optimizer

    DESCRIPTION: AdamW plus a cosine learning-rate schedule
    that steps once per batch.

    INPUTS: model, config, steps_per_epoch (int)

    OUTPUTS: (optimizer, scheduler)
    """
    optimizer = torch.optim.AdamW(params=model.parameters(),lr=config.lr, 
                                         weight_decay=config.weight_decay)
    t_max = config.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, t_max)

    return optimizer, scheduler


def use_bf16(device):
    """
    FUNCTION NAME: use_bf16

    DESCRIPTION: Whether bf16 autocast can be used.

    INPUTS: device (torch.device)

    OUTPUTS: bool
    """
    result: bool = False
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        result = True
    
    return result


def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, device, bf16):
    """
    FUNCTION NAME: train_one_epoch

    DESCRIPTION: One pass over the training data. The
    scheduler steps after every batch, and the mean loss is
    weighted by batch size so a short last batch doesn't
    skew it.

    INPUTS: model, loader, loss_fn, optimizer, scheduler,
    device, bf16 (bool)

    OUTPUTS: mean training loss (float)
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            logits = model(images)
        # Compute the loss in float32 for numerical stability
        loss = loss_fn(logits.float(), labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def evaluate(model, loader, loss_fn, device, bf16):
    """
    FUNCTION NAME: evaluate

    DESCRIPTION: Val loss and per tile-class accuracy. Uses
    raw logits (not model.predict) because BCEWithLogitsLoss
    expects logits.

    INPUTS: model, loader, loss_fn, device, bf16 (bool)

    OUTPUTS: (val_loss, val_accuracy)
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    total_decisions = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                logits = model(images)
            logits = logits.float()
            loss = loss_fn(logits, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions = torch.sigmoid(logits) >= PREDICTION_THRESHOLD
            correct += (predictions == labels.bool()).sum().item()
            total_decisions += labels.numel()

    val_loss = total_loss / max(total_samples, 1)
    val_accuracy = correct / max(total_decisions, 1)
    return val_loss, val_accuracy


def save_history(path, history):
    """
    FUNCTION NAME: save_history

    DESCRIPTION: Writes the epoch history to JSON.

    INPUTS: path (Path), history (list of EpochStats)

    OUTPUTS: None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump([asdict(stats) for stats in history], file, indent=2)


def history_path_for(out_path):
    """
    FUNCTION NAME: history_path_for

    DESCRIPTION: The history file that goes next to a
    checkpoint: checkpoints/model.pt ->
    checkpoints/model_history.json.

    INPUTS: out_path (str or Path)

    OUTPUTS: Path
    """
    out_path = Path(out_path)
    return out_path.with_name(f"{out_path.stem}_history.json")


def fit(config, verbose=True):
    """
    FUNCTION NAME: fit

    DESCRIPTION: Runs a full training job and saves the best
    checkpoint plus <name>_history.json.

    INPUTS: config (TrainConfig), verbose (bool)

    OUTPUTS: list of EpochStats
    """
    # 1. Check settings, seed everything, pick the device
    validate_config(config)
    set_seed(config.seed)
    device = select_device(config.device)
    bf16 = use_bf16(device)

    # 2. Data
    train_dataset, val_dataset = build_datasets(config)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Train and val datasets must both contain images.")
    train_loader, val_loader = build_loaders(config, train_dataset, val_dataset, device)
    classes = list(train_dataset.classes)

    # 3. Model with one output per class
    model = create_model(num_classes=len(classes), pretrained=config.pretrained).to(device)

    # 4-5. Multi-label loss, optimizer, and schedule
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer, scheduler = build_optimizer(model, config, len(train_loader))

    if verbose:
        print(f"Training on {device} (bf16={bf16}) with {len(train_dataset)} train / "
              f"{len(val_dataset)} val tiles, classes={classes}")

    # 6. Epoch loop; save a checkpoint whenever val loss improves
    out_path = Path(config.out_path)
    history = []
    best_val_loss = float("inf")
    best_epoch = None

    for epoch in range(1, config.epochs + 1):
        start = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler, device, bf16)
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device, bf16)
        stats = EpochStats(epoch=epoch,
                           train_loss=train_loss,
                           val_loss=val_loss,
                           val_accuracy=val_accuracy,
                           lr=lr,
                           seconds=time.perf_counter() - start)
        history.append(stats)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(out_path, model, classes,
                            extra={"config": asdict(config), "best_epoch": best_epoch})

        if verbose:
            marker = " *" if improved else ""
            print(f"epoch {epoch:3d}/{config.epochs}  train_loss {train_loss:.4f}  "
                  f"val_loss {val_loss:.4f}  val_acc {val_accuracy:.4f}  "
                  f"lr {lr:.2e}  {stats.seconds:.1f}s{marker}")

    # 7. Reload the best weights and resave with the full history
    best_model, _ = load_checkpoint(out_path, expected_classes=classes)
    history_dicts = [asdict(stats) for stats in history]
    save_checkpoint(out_path, best_model, classes,
                    extra={"config": asdict(config),
                           "best_epoch": best_epoch,
                           "history": history_dicts})

    # 8. History JSON next to the checkpoint
    history_path = history_path_for(out_path)
    save_history(history_path, history)

    if verbose:
        print(f"Best epoch {best_epoch} (val_loss {best_val_loss:.4f}). "
              f"Saved {out_path} and {history_path}")

    return history


def parse_args(argv=None):
    """
    FUNCTION NAME: parse_args

    DESCRIPTION: Parses command-line flags into a TrainConfig.

    INPUTS: argv (list of str, or None for sys.argv)

    OUTPUTS: TrainConfig
    """
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        prog="python -m Library.train",
        description="Fine-tune ResNet-18 on CAPTCHA tiles.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data", default=None,
                        help=f"dataset folder holding train/ and val/ (default: {defaults.data_dir})")
    source.add_argument("--synthetic", action="store_true",
                        help="train on generated placeholder tiles (wiring test only)")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size,
                        help="use 64 if the GPU runs out of memory")
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default=defaults.device)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--out", default=defaults.out_path,
                        help="checkpoint path; the history JSON is written next to it")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="start from random weights instead of ImageNet")
    args = parser.parse_args(argv)

    return TrainConfig(data_dir=args.data if args.data is not None else defaults.data_dir,
                       synthetic=args.synthetic,
                       epochs=args.epochs,
                       batch_size=args.batch_size,
                       lr=args.lr,
                       weight_decay=args.weight_decay,
                       device=args.device,
                       num_workers=args.num_workers,
                       seed=args.seed,
                       out_path=args.out,
                       pretrained=not args.no_pretrained)


def main(argv=None):
    """
    FUNCTION NAME: main

    DESCRIPTION: Command-line entry point.

    INPUTS: argv (list of str, or None)

    OUTPUTS: exit code (int)
    """
    config = parse_args(argv)
    try:
        fit(config)
    except torch.cuda.OutOfMemoryError:
        print(f"Out of GPU memory at batch size {config.batch_size}. "
              "Try again with --batch-size 64.", file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
