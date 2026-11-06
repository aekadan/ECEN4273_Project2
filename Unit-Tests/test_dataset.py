
import torch

from Library.dataset import SyntheticTileDataset


def test_synthetic_dataset_length():
    dataset = SyntheticTileDataset(num_samples=12)

    assert len(dataset) == 12


def test_synthetic_dataset_image_shape():
    dataset = SyntheticTileDataset(num_samples=9)

    image, labels = dataset[0]

    assert image.shape == (3, 224, 224)
    assert image.dtype == torch.float32


def test_synthetic_dataset_label_shape():
    dataset = SyntheticTileDataset(num_samples=9)

    image, labels = dataset[0]

    assert labels.shape == (4,)
    assert labels.dtype == torch.float32


def test_synthetic_dataset_labels():
    dataset = SyntheticTileDataset(num_samples=9)

    for index in range(len(dataset)):
        image, labels = dataset[index]

        expected_class = index % len(dataset.classes)

        assert labels.sum().item() == 1
        assert labels[expected_class].item() == 1


def test_synthetic_dataset_reproducibility():
    dataset = SyntheticTileDataset(
        num_samples=9,
        seed=42
    )

    image1, labels1 = dataset[0]
    image2, labels2 = dataset[0]

    assert torch.equal(image1, image2)
    assert torch.equal(labels1, labels2)

def test_multilabel_csv(tmp_path):
    from PIL import Image
    from Library.dataset import TileDataset

    image_path = tmp_path / "tile1.png"
    Image.new("RGB", (100, 100), color="blue").save(image_path)

    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "filename,labels\n"
        "tile1.png,bus;bicycle\n",
        encoding="utf-8"
    )

    dataset = TileDataset(
        str(tmp_path),
        classes=["bus", "bicycle", "traffic_light", "hydrant"]
    )

    image, labels = dataset[0]

    assert image.shape == (3, 224, 224)
    assert labels.tolist() == [1.0, 1.0, 0.0, 0.0]

def test_background_image_csv(tmp_path):
    from PIL import Image
    from Library.dataset import TileDataset

    image_path = tmp_path / "background.png"
    Image.new("RGB", (100, 100), color="white").save(image_path)

    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "filename,labels\n"
        "background.png,\n",
        encoding="utf-8"
    )

    dataset = TileDataset(
        str(tmp_path),
        classes=["bus", "bicycle", "traffic_light", "hydrant"]
    )

    image, labels = dataset[0]

    assert image.shape == (3, 224, 224)
    assert labels.tolist() == [0.0, 0.0, 0.0, 0.0]
