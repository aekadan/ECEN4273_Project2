
#############################################################
#   FILE NAME: prepare_dataset.py
#
#   DESCRIPTION: This file verifies and inspects the COCO
#   image dataset, records category and licensing information,
#   and checks the prepared dataset directories.
#
#   Authors: Group 2
#############################################################

#############################################################
#   Libraries
#############################################################

from pathlib import Path
from collections import Counter
from PIL import Image
import json
import zipfile
import csv
import random
import io
from collections import defaultdict

#############################################################
#   Constants
#############################################################

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "Image-Data"

TRAIN_ZIP = DATA_DIR / "original" / "train2017.zip"

ANNOTATIONS = (
    Path.home()
    / "fiftyone"
    / "coco-2017"
    / "raw"
    / "instances_train2017.json"
)

#############################################################
#   Dataset Configuration
#############################################################

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

#############################################################
#   FUNCTIONS
#############################################################


def verify_sources():
    """
    FUNCTION NAME: verify_sources

    DESCRIPTION:
        Verifies that the original COCO archive and
        annotation file exist.

    INPUTS:
        None

    OUTPUTS:
        None
    """

    if not TRAIN_ZIP.is_file():
        raise FileNotFoundError(TRAIN_ZIP)

    if not ANNOTATIONS.is_file():
        raise FileNotFoundError(ANNOTATIONS)

    if not zipfile.is_zipfile(TRAIN_ZIP):
        raise ValueError("Invalid ZIP archive.")

    print("Source files verified.")


def load_annotations():
    """
    FUNCTION NAME: load_annotations

    DESCRIPTION:
        Loads the COCO JSON annotation file.

    INPUTS:
        None

    OUTPUTS:
        data: COCO annotation dictionary.
    """

    with ANNOTATIONS.open(encoding="utf-8") as file:
        return json.load(file)


def inspect_archive():
    """
    FUNCTION NAME: inspect_archive

    DESCRIPTION:
        Counts photographs stored in the original archive.

    INPUTS:
        None

    OUTPUTS:
        count: Number of photographs.
    """

    with zipfile.ZipFile(TRAIN_ZIP) as archive:
        images = [
            name for name in archive.namelist()
            if Path(name).suffix.lower() in IMAGE_EXTENSIONS
        ]

    return len(images)


def export_category_statistics(data):
    """
    FUNCTION NAME: export_category_statistics

    DESCRIPTION:
        Records the annotation count for each COCO category.

    INPUTS:
        data: COCO annotation dictionary.

    OUTPUTS:
        category_statistics.csv
    """

    counts = Counter(
        annotation["category_id"]
        for annotation in data["annotations"]
    )

    output = DATA_DIR / "category_statistics.csv"

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "category_id",
            "category_name",
            "annotation_count"
        ])

        for category in data["categories"]:
            writer.writerow([
                category["id"],
                category["name"],
                counts[category["id"]]
            ])

    print("Category statistics saved:", output)


def export_license_information(data):
    """
    FUNCTION NAME: export_license_information

    DESCRIPTION:
        Records the original image license information
        and source URLs.

    INPUTS:
        data: COCO annotation dictionary.

    OUTPUTS:
        source_licenses.csv
    """

    licenses = {
        license_info["id"]: license_info
        for license_info in data.get("licenses", [])
    }

    output = DATA_DIR / "source_licenses.csv"

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "image_id",
            "filename",
            "license_id",
            "license_name",
            "license_url",
            "source_url"
        ])

        for image in data["images"]:
            license_id = image.get("license")
            info = licenses.get(license_id, {})

            writer.writerow([
                image["id"],
                image["file_name"],
                license_id,
                info.get("name", ""),
                info.get("url", ""),
                image.get("flickr_url", "")
            ])

    print("License information saved:", output)


def verify_output_directories():
    """
    FUNCTION NAME: verify_output_directories

    DESCRIPTION:
        Verifies the training, validation, and testing
        directories and checks existing image integrity.

    INPUTS:
        None

    OUTPUTS:
        None
    """

    print("\nDataset directory inspection:")

    for directory in [TRAIN_DIR, VAL_DIR, TEST_DIR]:

        directory.mkdir(parents=True, exist_ok=True)

        images = [
            path for path in directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        valid = 0
        invalid = 0

        for path in images:
            try:
                with Image.open(path) as image:
                    image.verify()

                valid += 1

            except Exception:
                invalid += 1
                print("Invalid image:", path)

        print(
            f"{directory.name}: "
            f"{valid} valid images, "
            f"{invalid} invalid images"
        )

def crop_image(image, bbox):
    """
    FUNCTION NAME: crop_image

    DESCRIPTION:
        Extracts a rectangular region from an image.

    INPUTS:
        image: PIL image.
        bbox: Bounding box coordinates.

    OUTPUTS:
        Cropped PIL image.
    """

    x, y, width, height = bbox

    return image.crop((
        x,
        y,
        x + width,
        y + height
    ))

def get_valid_annotations(data):
    """
    FUNCTION NAME: get_valid_annotations

    DESCRIPTION:
        Identifies annotations with valid image references
        and positive bounding box dimensions.

    INPUTS:
        data: COCO annotation dictionary.

    OUTPUTS:
        valid_annotations: List of valid annotations.
    """

    image_ids = {
        image["id"] for image in data["images"]
    }

    valid_annotations = []

    for annotation in data["annotations"]:

        bbox = annotation.get("bbox", [])

        if len(bbox) != 4:
            continue

        x, y, width, height = bbox

        if width <= 0 or height <= 0:
            continue

        if annotation["image_id"] not in image_ids:
            continue

        valid_annotations.append(annotation)

    return valid_annotations

def prepare_object_dataset(data, categories, samples_per_class):
    """
    FUNCTION NAME: prepare_object_dataset

    DESCRIPTION:
        Prepares a balanced object classification dataset
        from COCO photographs and bounding box annotations.

    INPUTS:
        data: COCO annotation dictionary.
        categories: Dictionary of category names and IDs.
        samples_per_class: Requested samples per category.

    OUTPUTS:
        Prepared image directories and source manifest.
    """

    rng = random.Random(RANDOM_SEED)

    images = {
        image["id"]: image
        for image in data["images"]
    }

    annotations = get_valid_annotations(data)

    grouped = defaultdict(list)

    for annotation in annotations:
        if not annotation.get("iscrowd", 0):
            grouped[annotation["category_id"]].append(
                annotation
            )

    used_images = set()

    manifest = []

    with zipfile.ZipFile(TRAIN_ZIP) as archive:

        for category, category_id in categories.items():

            candidates = grouped[category_id].copy()
            rng.shuffle(candidates)

            selected = 0

            for annotation in candidates:

                if selected >= samples_per_class:
                    break

                image_id = annotation["image_id"]

                if image_id in used_images:
                    continue

                metadata = images[image_id]

                archive_name = (
                    "train2017/" + metadata["file_name"]
                )

                try:
                    with Image.open(
                        io.BytesIO(
                            archive.read(archive_name)
                        )
                    ) as original:

                        x, y, width, height = annotation["bbox"]

                        if (
                            x < 0
                            or y < 0
                            or x + width > original.width
                            or y + height > original.height
                        ):
                            continue

                        cropped = crop_image(
                            original.convert("RGB"),
                            annotation["bbox"]
                        )

                        if min(cropped.size) < 1:
                            continue

                        if selected < int(
                            samples_per_class * TRAIN_RATIO
                        ):
                            directory = TRAIN_DIR

                        elif selected < int(
                            samples_per_class
                            * (TRAIN_RATIO + VAL_RATIO)
                        ):
                            directory = VAL_DIR

                        else:
                            directory = TEST_DIR

                        output_dir = directory / category

                        output_dir.mkdir(
                            parents=True,
                            exist_ok=True
                        )

                        filename = (
                            f"{image_id}_"
                            f"{annotation['id']}.jpg"
                        )

                        cropped.save(
                            output_dir / filename,
                            quality=95
                        )

                        manifest.append([
                            directory.name,
                            category,
                            filename,
                            image_id,
                            annotation["id"],
                            metadata.get("license"),
                            metadata.get("flickr_url", "")
                        ])

                        used_images.add(image_id)

                        selected += 1

                except (OSError, KeyError) as error:
                    print("Skipped image:", image_id, error)

            print(
                f"{category}: "
                f"{selected}/{samples_per_class} images"
            )

            if selected < samples_per_class:
                print(
                    "Warning: Requested sample count "
                    "was not reached."
                )

    output = DATA_DIR / "dataset_manifest.csv"

    with output.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "split",
            "category",
            "filename",
            "source_image_id",
            "annotation_id",
            "license_id",
            "source_url"
        ])

        writer.writerows(manifest)

    print("Dataset manifest saved:", output)


def verify_class_counts():
    """
    FUNCTION NAME: verify_class_counts

    DESCRIPTION:
        Reports the number of prepared images in each
        dataset split and category.

    INPUTS:
        None

    OUTPUTS:
        Dataset image counts.
    """

    total = 0

    for directory in [TRAIN_DIR, VAL_DIR, TEST_DIR]:

        print(f"\n{directory.name.upper()}")

        for category_dir in sorted(directory.iterdir()):

            if not category_dir.is_dir():
                continue

            count = sum(
                1 for path in category_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            )

            total += count

            print(f"{category_dir.name}: {count}")

    print("\nTotal prepared images:", total)

def main():
    """
    FUNCTION NAME: main

    DESCRIPTION:
        Verifies source files, prepares the image dataset,
        generates documentation, and validates the
        completed dataset.

    INPUTS:
        None

    OUTPUTS:
        Prepared dataset and verification reports.
    """

    verify_sources()

    data = load_annotations()
    valid_annotations = get_valid_annotations(data)

    print("Photographs:", inspect_archive())
    print("Annotations:", len(data["annotations"]))
    print("Categories:", len(data["categories"]))
    print("Valid annotations:", len(valid_annotations))
    print(
        "Rejected annotations:",
        len(data["annotations"]) - len(valid_annotations)
    )

    export_category_statistics(data)
    export_license_information(data)

    categories = {
        "hydrant": 11,
        "bicycle": 2,
        "bus": 6,
        "traffic_light": 10
    }

    samples_per_class = 1000

    print("\nPreparing image dataset...")

    prepare_object_dataset(
        data,
        categories,
        samples_per_class
    )

    verify_output_directories()
    verify_class_counts()

    print("\nDataset preparation complete.")

#############################################################
#   MAIN
#############################################################

if __name__ == "__main__":
    main()
