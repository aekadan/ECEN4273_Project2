
import csv
import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from Library.tile_sources import SyntheticTileSource


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


class TileDataset(Dataset):
    def __init__(self, data_dir, classes=None):
        self.data_dir = data_dir
        self.transform = get_transform()
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
        return len(self.samples)

    def __getitem__(self, index):
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
    def __init__(self, num_samples=100, seed=42):
        self.source = SyntheticTileSource()
        self.classes = self.source.classes

        self.class_to_idx = {
            name: idx for idx, name in enumerate(self.classes)
        }

        self.num_samples = num_samples
        self.seed = seed
        self.transform = get_transform()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
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
