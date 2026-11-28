
#############################################################
#   FILE NAME: dataset.py
#
#   DESCRIPTION: This file loads and prepares image tiles
#   for the CAPTCHA recognition model. It supports loading
#   images from folders, reading multiple labels from CSV
#   files, and generating synthetic CAPTCHA image tiles.
#
#   Authors: Group 2
#############################################################

#############################################################
#   Libraries
#############################################################

import csv
import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

#############################################################
#   Files Import
#############################################################

from Library.tile_sources import SyntheticTileSource

#############################################################
#   Global Constants
#############################################################

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

#############################################################
#   FUNCTIONS
#############################################################

def get_transform():
    """
    FUNCTION NAME: get_transform

    DESCRIPTION: This function prepares images for the
    ResNet-18 model by resizing them to 224x224, converting
    them into tensors, and normalizing the image values.

    INPUTS: None

    OUTPUTS: Image transformation
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])


#############################################################
#   CLASSES
#############################################################

class TileDataset(Dataset):
    """
    CLASS NAME: TileDataset

    DESCRIPTION: This class loads image tiles from a dataset
    folder and prepares their labels for the AI model.
    Images can be organized into class folders or labeled
    using a CSV file that supports multiple classes.

    INPUTS: data_dir, classes, transform

    OUTPUTS: Image tensor and label tensor
    """

    def __init__(self, data_dir, classes=None, transform=None):
        """
        FUNCTION NAME: __init__

        DESCRIPTION: This function initializes the dataset,
        determines whether to load images from folders or
        a CSV file, and creates the class index mapping.
        If no transform is given, get_transform() is used.

        INPUTS: data_dir, classes, transform (optional
        image transform, e.g. training augmentation)

        OUTPUTS: None
        """
        self.data_dir = data_dir
        self.transform = (
            transform if transform is not None else get_transform()
        )
        self.samples = []

        csv_path = os.path.join(data_dir, "labels.csv")

        if os.path.isfile(csv_path):
            self._load_csv(csv_path, classes)
        else:
            self._load_folders(classes)

        self.class_to_idx = {
            name: idx for idx, name in enumerate(self.classes)
        }

    def _load_folders(self, classes):
        """
        FUNCTION NAME: _load_folders

        DESCRIPTION: This function loads image file paths
        from class folders and assigns a class label to
        each image based on its folder name.

        INPUTS: classes

        OUTPUTS: None
        """
        folders = sorted([
            name for name in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, name))
        ])

        self.classes = list(classes) if classes is not None else folders

        for class_name in folders:
            if class_name not in self.classes:
                raise ValueError(
                    f"Unknown class folder: {class_name}"
                )

            class_dir = os.path.join(self.data_dir, class_name)

            for filename in sorted(os.listdir(class_dir)):
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_path = os.path.join(class_dir, filename)
                    self.samples.append((image_path, [class_name]))

    def _load_csv(self, csv_path, classes):
        """
        FUNCTION NAME: _load_csv

        DESCRIPTION: This function reads image labels from
        a CSV file. It supports multiple labels per image
        and images that do not contain any target classes.

        INPUTS: csv_path, classes

        OUTPUTS: None
        """
        with open(csv_path, newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)

            if reader.fieldnames is None:
                raise ValueError("labels.csv is empty.")

            if not {"filename", "labels"}.issubset(reader.fieldnames):
                raise ValueError(
                    "labels.csv must contain filename and labels columns."
                )

            rows = list(reader)

        discovered_classes = set()

        for row in rows:
            filename = row["filename"].strip()
            label_text = (row["labels"] or "").strip()

            image_labels = [
                label.strip()
                for label in label_text.split(";")
                if label.strip()
            ]

            image_path = os.path.join(self.data_dir, filename)

            if not os.path.isfile(image_path):
                raise FileNotFoundError(image_path)

            discovered_classes.update(image_labels)
            self.samples.append((image_path, image_labels))

        self.classes = (
            list(classes)
            if classes is not None
            else sorted(discovered_classes)
        )

        unknown = discovered_classes.difference(self.classes)

        if unknown:
            raise ValueError(
                f"Unknown classes in labels.csv: {sorted(unknown)}"
            )

    def __len__(self):
        """
        FUNCTION NAME: __len__

        DESCRIPTION: This function returns the total number
        of image samples stored in the dataset.

        INPUTS: None

        OUTPUTS: Number of samples
        """
        return len(self.samples)

    def __getitem__(self, index):
        """
        FUNCTION NAME: __getitem__

        DESCRIPTION: This function loads an image at the
        specified index, applies the image transformations,
        and creates a label tensor for its classes.

        INPUTS: index

        OUTPUTS: Image tensor and label tensor
        """
        image_path, image_labels = self.samples[index]

        with Image.open(image_path) as image:
            image = self.transform(image.convert("RGB"))

        labels = torch.zeros(
            len(self.classes),
            dtype=torch.float32
        )

        for class_name in image_labels:
            labels[self.class_to_idx[class_name]] = 1.0

        return image, labels


class SyntheticTileDataset(Dataset):
    """
    CLASS NAME: SyntheticTileDataset

    DESCRIPTION: This class generates synthetic CAPTCHA
    image tiles using the SyntheticTileSource. Each image
    is transformed and assigned a class label for testing
    and training the AI model.

    INPUTS: num_samples, seed, transform

    OUTPUTS: Image tensor and label tensor
    """

    def __init__(self, num_samples=100, seed=42, transform=None):
        """
        FUNCTION NAME: __init__

        DESCRIPTION: This function initializes the synthetic
        dataset, sets the number of samples and random seed,
        and creates the class index mapping. If no transform
        is given, get_transform() is used.

        INPUTS: num_samples, seed, transform (optional image
        transform, e.g. training augmentation)

        OUTPUTS: None
        """
        self.source = SyntheticTileSource()
        self.classes = self.source.classes

        self.class_to_idx = {
            name: idx for idx, name in enumerate(self.classes)
        }

        self.num_samples = num_samples
        self.seed = seed
        self.transform = (
            transform if transform is not None else get_transform()
        )

    def __len__(self):
        """
        FUNCTION NAME: __len__

        DESCRIPTION: This function returns the total number
        of synthetic image samples in the dataset.

        INPUTS: None

        OUTPUTS: Number of samples
        """
        return self.num_samples

    def __getitem__(self, index):
        """
        FUNCTION NAME: __getitem__

        DESCRIPTION: This function generates a synthetic
        CAPTCHA image, applies the image transformations,
        and creates a label tensor for the selected class.

        INPUTS: index

        OUTPUTS: Image tensor and label tensor
        """
        if index < 0 or index >= self.num_samples:
            raise IndexError("Sample index is out of range.")

        rng = random.Random(self.seed + index)

        label = self.classes[index % len(self.classes)]

        image = self.source.get_tile(label, rng)
        image = self.transform(image)

        labels = torch.zeros(
            len(self.classes),
            dtype=torch.float32
        )

        labels[self.class_to_idx[label]] = 1.0

        return image, labels
