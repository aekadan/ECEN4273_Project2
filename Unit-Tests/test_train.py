#############################################################
#   FILE NAME: test_train.py
#
#   DESCRIPTION: This file contains unit tests for the
#   training loop (Library/train.py). Every test runs on the
#   CPU with tiny synthetic or temporary folder datasets, so
#   no GPU or weight download is needed.
#
#   Covers every function in train.py (config, data,
#   optimizer, training/eval loops, fit, the command line)
#   and the checkpoint helpers in model.py.
#
#   Authors: Group 2
#############################################################

#############################################################
#   Libraries
#############################################################

import json
import random

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

#############################################################
#   Files Import
#############################################################

from Library import train
from Library import model as model_lib
from Library.dataset import IMAGE_SIZE

#############################################################
#   Global Constants
#############################################################

TINY_TRAIN_SAMPLES = 16
TINY_VAL_SAMPLES = 8

#############################################################
#   FUNCTIONS
#############################################################

def make_config(tmp_path, **overrides):
    """
    FUNCTION NAME: make_config

    DESCRIPTION: Builds a small TrainConfig that runs on the
    CPU with no workers, no pretrained download, and a
    checkpoint path inside tmp_path. Keyword arguments
    override any field.

    INPUTS: tmp_path (pathlib.Path), **overrides (TrainConfig
    fields)

    OUTPUTS: TrainConfig
    """
    settings = dict(
        synthetic=True,
        epochs=1,
        batch_size=8,
        device="cpu",
        num_workers=0,
        pretrained=False,
        out_path=str(tmp_path / "model.pt"),
        synthetic_train_samples=TINY_TRAIN_SAMPLES,
        synthetic_val_samples=TINY_VAL_SAMPLES,
    )
    settings.update(overrides)
    return train.TrainConfig(**settings)


def make_image_folders(root, layout):
    """
    FUNCTION NAME: make_image_folders

    DESCRIPTION: Writes a tiny folder dataset of solid-color
    PNGs. layout maps a split name to a list of class names,
    and each class folder gets one image.

    INPUTS: root (pathlib.Path), layout (dict of str -> list
    of str), e.g. {"train": ["apple", "zebra"], "val": ["zebra"]}

    OUTPUTS: None
    """
    for split, class_names in layout.items():
        for class_name in class_names:
            folder = root / split / class_name
            folder.mkdir(parents=True)
            Image.new("RGB", (50, 50), (120, 60, 30)).save(folder / "tile.png")

def make_loader(inputs, labels, batch_size):
    """
    FUNCTION NAME: make_loader

    DESCRIPTION: Wraps tensors in an unshuffled DataLoader
    for testing the train/eval loops with tiny models.

    INPUTS: inputs (tensor), labels (tensor), batch_size (int)

    OUTPUTS: DataLoader
    """
    return DataLoader(TensorDataset(inputs, labels), batch_size=batch_size, shuffle=False)


class ZeroModel(torch.nn.Module):
    """
    CLASS NAME: ZeroModel

    DESCRIPTION: Always outputs logits of 0 (probability 0.5)
    for num_classes classes, so losses and accuracy have
    exact known values.

    INPUTS: num_classes (int)

    OUTPUTS: None
    """

    def __init__(self, num_classes):
        """
        FUNCTION NAME: __init__

        DESCRIPTION: Stores the class count and a dummy
        parameter.

        INPUTS: num_classes (int)

        OUTPUTS: None
        """
        super().__init__()
        self.num_classes = num_classes
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        """
        FUNCTION NAME: forward

        DESCRIPTION: Returns all-zero logits.

        INPUTS: inputs (tensor, batch first)

        OUTPUTS: N x num_classes tensor
        """
        return torch.zeros(inputs.size(0), self.num_classes) + 0 * self.dummy

#############################################################
#   CLASSES
#############################################################

class TestTrain:
    """
    CLASS NAME: TestTrain

    DESCRIPTION: This class contains unit tests for the
    implemented parts of Library/train.py.
    """

    def test_train_config_defaults(self):
        """
        FUNCTION NAME: test_train_config_defaults

        DESCRIPTION: Checks TrainConfig's defaults match the
        module constants.

        INPUTS: None

        OUTPUTS: None
        """
        config = train.TrainConfig()
        assert config.synthetic is False
        assert config.epochs == train.DEFAULT_EPOCHS
        assert config.batch_size == train.DEFAULT_BATCH_SIZE
        assert config.lr == train.DEFAULT_LR
        assert config.device == "auto"
        assert config.seed == train.DEFAULT_SEED
        assert config.out_path == train.DEFAULT_CHECKPOINT
        assert config.pretrained is True


    def test_epoch_stats_ignores_seconds(self):
        """
        FUNCTION NAME: test_epoch_stats_ignores_seconds

        DESCRIPTION: Two EpochStats with the same scores but
        different timings compare equal.

        INPUTS: None

        OUTPUTS: None
        """
        first = train.EpochStats(1, 0.5, 0.4, 0.9, 1e-4, seconds=1.0)
        second = train.EpochStats(1, 0.5, 0.4, 0.9, 1e-4, seconds=99.0)
        assert first == second


    def test_get_train_transform(self):
        """
        FUNCTION NAME: test_get_train_transform

        DESCRIPTION: The train transform turns a PIL image of
        any size into a normalized 3x224x224 float tensor, is
        random from call to call, and is repeatable under the
        same torch seed.

        INPUTS: None

        OUTPUTS: None
        """
        transform = train.get_train_transform()
        image = Image.new("RGB", (100, 80))
        image.putdata([(x % 256, y % 256, (x * y) % 256)
                       for y in range(80) for x in range(100)])

        torch.manual_seed(0)
        first = transform(image)
        second = transform(image)
        torch.manual_seed(0)
        repeat = transform(image)

        assert first.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
        assert first.dtype == torch.float32
        assert first.min() < 0  # normalized, not plain 0-1 values
        assert not torch.equal(first, second)
        assert torch.equal(first, repeat)


    def test_select_device(self, monkeypatch):
        """
        FUNCTION NAME: test_select_device

        DESCRIPTION: Checks "cpu", "auto", and "cuda" with and
        without a GPU (faked with monkeypatch), and rejects
        unknown names.

        INPUTS: monkeypatch (pytest fixture)

        OUTPUTS: None
        """
        assert train.select_device("cpu") == torch.device("cpu")

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert train.select_device("auto") == torch.device("cpu")
        with pytest.raises(ValueError):
            train.select_device("cuda")

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert train.select_device("auto") == torch.device("cuda")
        assert train.select_device("cuda") == torch.device("cuda")

        with pytest.raises(ValueError):
            train.select_device("gpu")


    def test_set_seed(self):
        """
        FUNCTION NAME: test_set_seed

        DESCRIPTION: The same seed gives the same Python and
        torch random numbers; a different seed does not.

        INPUTS: None

        OUTPUTS: None
        """
        train.set_seed(123)
        first = (random.random(), torch.rand(5))
        train.set_seed(123)
        repeat = (random.random(), torch.rand(5))
        train.set_seed(456)
        other = (random.random(), torch.rand(5))

        assert first[0] == repeat[0]
        assert torch.equal(first[1], repeat[1])
        assert first[0] != other[0]
        assert not torch.equal(first[1], other[1])


    def test_validate_config_accepts_good_config(self, tmp_path):
        """
        FUNCTION NAME: test_validate_config_accepts_good_config

        DESCRIPTION: A valid config (and the defaults) pass
        without raising.

        INPUTS: tmp_path (pytest fixture)

        OUTPUTS: None
        """
        train.validate_config(make_config(tmp_path))
        train.validate_config(train.TrainConfig())


    @pytest.mark.parametrize("field, value", [
        ("epochs", 0),
        ("epochs", 1.5),
        ("batch_size", 0),
        ("batch_size", 8.0),
        ("lr", 0),
        ("lr", -1e-3),
        ("weight_decay", -1e-4),
        ("num_workers", -1),
        ("synthetic_train_samples", 0),
        ("synthetic_val_samples", 0),
    ])
    def test_validate_config_rejects_bad_values(self, tmp_path, field, value):
        """
        FUNCTION NAME: test_validate_config_rejects_bad_values

        DESCRIPTION: Each bad setting raises ValueError.

        INPUTS: tmp_path (pytest fixture), field (str), value

        OUTPUTS: None
        """
        config = make_config(tmp_path, **{field: value})
        with pytest.raises(ValueError):
            train.validate_config(config)


    def test_build_datasets_synthetic(self, tmp_path):
        """
        FUNCTION NAME: test_build_datasets_synthetic

        DESCRIPTION: Synthetic mode returns train and val sets
        of the configured sizes, with the augmenting transform
        on train, the plain one on val, and a separate val seed.
        It never looks at data_dir.

        INPUTS: tmp_path (pytest fixture)

        OUTPUTS: None
        """
        config = make_config(tmp_path, data_dir=str(tmp_path / "missing"))
        train_set, val_set = train.build_datasets(config)

        assert len(train_set) == TINY_TRAIN_SAMPLES
        assert len(val_set) == TINY_VAL_SAMPLES
        assert train_set.classes == val_set.classes

        image, label = train_set[0]
        assert image.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
        assert label.shape == (len(train_set.classes),)

        assert isinstance(train_set.transform.transforms[0],
                          transforms.RandomResizedCrop)
        assert isinstance(val_set.transform.transforms[0],
                          transforms.Resize)
        assert val_set.seed == config.seed + train.SYNTHETIC_VAL_SEED_OFFSET


    def test_build_datasets_folders(self, tmp_path):
        """
        FUNCTION NAME: test_build_datasets_folders

        DESCRIPTION: Folder mode reads train/ and val/, and
        val uses train's class list so label positions match
        even when val is missing a class.

        INPUTS: tmp_path (pytest fixture)

        OUTPUTS: None
        """
        make_image_folders(tmp_path, {"train": ["apple", "zebra"],
                                      "val": ["zebra"],
                                      "test": ["kiwi"]})
        config = make_config(tmp_path, synthetic=False,
                             data_dir=str(tmp_path))
        train_set, val_set = train.build_datasets(config)

        assert len(train_set) == 2
        assert len(val_set) == 1
        assert train_set.classes == ["apple", "zebra"]
        assert val_set.classes == train_set.classes
        assert val_set[0][1].tolist() == [0.0, 1.0]

        assert isinstance(train_set.transform.transforms[0],
                          transforms.RandomResizedCrop)
        assert isinstance(val_set.transform.transforms[0],
                          transforms.Resize)


    @pytest.mark.parametrize("missing_split", ["train", "val"])
    def test_build_datasets_missing_split(self, tmp_path, missing_split):
        """
        FUNCTION NAME: test_build_datasets_missing_split

        DESCRIPTION: A missing train/ or val/ folder raises
        FileNotFoundError naming that folder.

        INPUTS: tmp_path (pytest fixture), missing_split (str)

        OUTPUTS: None
        """
        present = "val" if missing_split == "train" else "train"
        make_image_folders(tmp_path, {present: ["apple"]})
        config = make_config(tmp_path, synthetic=False,
                             data_dir=str(tmp_path))

        with pytest.raises(FileNotFoundError, match=missing_split):
            train.build_datasets(config)

    def test_build_datasets_unknown_val_class(self, tmp_path):
        """
        FUNCTION NAME: test_build_datasets_unknown_val_class

        DESCRIPTION: A class that appears in val/ but not in
        train/ raises ValueError.

        INPUTS: tmp_path (pytest fixture)

        OUTPUTS: None
        """
        make_image_folders(tmp_path, {"train": ["apple"],
                                      "val": ["apple", "kiwi"]})
        config = make_config(tmp_path, synthetic=False,
                             data_dir=str(tmp_path))

        with pytest.raises(ValueError):
            train.build_datasets(config)

    def test_build_loaders_batches(self, tmp_path):
        """
        FUNCTION NAME: test_build_loaders_batches

        DESCRIPTION: Both loaders yield batches of the
        configured size with image and multi-hot label
        tensors, and cover every sample once per epoch.

        INPUTS: tmp_path (pytest fixture)

        OUTPUTS: None
        """
        config = make_config(tmp_path)
        train_set, val_set = train.build_datasets(config)
        train_loader, val_loader = train.build_loaders(
            config, train_set, val_set, torch.device("cpu"))

        images, labels = next(iter(train_loader))
        assert images.shape == (config.batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
        assert labels.shape == (config.batch_size, len(train_set.classes))

        assert sum(len(batch[0]) for batch in train_loader) == TINY_TRAIN_SAMPLES
        assert sum(len(batch[0]) for batch in val_loader) == TINY_VAL_SAMPLES

    def test_build_loaders_shuffle(self, tmp_path):
        """
        FUNCTION NAME: test_build_loaders_shuffle

        DESCRIPTION: Train shuffles in an order fixed by the
        config seed; val keeps the dataset order. Uses a plain
        list of numbers as the dataset so the order is easy to
        read.

        INPUTS: tmp_path (pytest fixture)

        OUTPUTS: None
        """
        data = list(range(32))
        device = torch.device("cpu")

        def train_order(seed):
            config = make_config(tmp_path, batch_size=32, seed=seed)
            loader, _ = train.build_loaders(config, data, data, device)
            return next(iter(loader)).tolist()

        assert train_order(0) != data
        assert train_order(0) == train_order(0)
        assert train_order(0) != train_order(1)

        config = make_config(tmp_path, batch_size=32)
        _, val_loader = train.build_loaders(config, data, data, device)
        assert next(iter(val_loader)).tolist() == data

    def test_build_loaders_options(self, tmp_path):
        """
        FUNCTION NAME: test_build_loaders_options

        DESCRIPTION: pin_memory is only on for CUDA, and
        workers are passed through (kept alive between epochs
        only when there are any).

        INPUTS: tmp_path (pytest fixture)

        OUTPUTS: None
        """
        data = list(range(4))

        config = make_config(tmp_path)
        train_loader, val_loader = train.build_loaders(
            config, data, data, torch.device("cpu"))
        for loader in (train_loader, val_loader):
            assert loader.pin_memory is False
            assert loader.num_workers == 0
            assert loader.persistent_workers is False

        config = make_config(tmp_path, num_workers=2)
        train_loader, val_loader = train.build_loaders(
            config, data, data, torch.device("cuda"))
        for loader in (train_loader, val_loader):
            assert loader.pin_memory is True
            assert loader.num_workers == 2
            assert loader.persistent_workers is True

    def test_build_optimizer_settings(self, tmp_path):
        """
        FUNCTION NAME: test_build_optimizer_settings

        DESCRIPTION: Returns an AdamW optimizer that holds
        every model parameter with the config's lr and weight
        decay, and a cosine scheduler whose length is the
        total number of batches (epochs * steps_per_epoch).

        INPUTS: tmp_path (pytest fixture)

        OUTPUTS: None
        """
        model = torch.nn.Linear(4, 2)
        config = make_config(tmp_path, epochs=3, lr=1e-3, weight_decay=0.05)
        optimizer, scheduler = train.build_optimizer(model, config,
                                                     steps_per_epoch=7)

        assert isinstance(optimizer, torch.optim.AdamW)
        group = optimizer.param_groups[0]
        assert group["lr"] == pytest.approx(1e-3)
        assert group["weight_decay"] == pytest.approx(0.05)
        optimized = {id(p) for g in optimizer.param_groups for p in g["params"]}
        assert optimized == {id(p) for p in model.parameters()}

        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert scheduler.T_max == 3 * 7

    def test_build_optimizer_cosine_schedule(self, tmp_path):
        """
        FUNCTION NAME: test_build_optimizer_cosine_schedule

        DESCRIPTION: Stepping once per batch, the learning rate
        starts at config.lr, is half of it at the midpoint,
        keeps falling, and reaches about 0 at the last step.

        INPUTS: tmp_path (pytest fixture)

        OUTPUTS: None
        """
        model = torch.nn.Linear(4, 2)
        config = make_config(tmp_path, epochs=2, lr=3e-4)
        optimizer, scheduler = train.build_optimizer(model, config,
                                                     steps_per_epoch=5)

        lrs = [scheduler.get_last_lr()[0]]
        for _ in range(10):
            optimizer.step()
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])

        assert lrs[0] == pytest.approx(3e-4)
        assert lrs[5] == pytest.approx(1.5e-4)
        assert lrs[10] == pytest.approx(0.0, abs=1e-12)
        assert all(later < earlier for earlier, later in zip(lrs, lrs[1:]))

    def test_build_optimizer_updates_weights(self, tmp_path):
        """
        FUNCTION NAME: test_build_optimizer_updates_weights

        DESCRIPTION: One optimizer step after a backward pass
        changes the model's weights.

        INPUTS: tmp_path (pytest fixture)

        OUTPUTS: None
        """
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 2)
        before = model.weight.detach().clone()
        optimizer, _ = train.build_optimizer(model, make_config(tmp_path),
                                             steps_per_epoch=1)

        loss = model(torch.randn(8, 4)).pow(2).mean()
        loss.backward()
        optimizer.step()

        assert not torch.equal(model.weight, before)

    @pytest.mark.parametrize("supported", [True, False])
    def test_use_bf16_cuda(self, monkeypatch, supported):
        """
        FUNCTION NAME: test_use_bf16_cuda

        DESCRIPTION: On a CUDA device, bf16 is used exactly
        when the GPU supports it (faked with monkeypatch so
        the test runs without a GPU).

        INPUTS: monkeypatch (pytest fixture), supported (bool)

        OUTPUTS: None
        """
        monkeypatch.setattr(torch.cuda, "is_bf16_supported",
                            lambda *args, **kwargs: supported)
        assert train.use_bf16(torch.device("cuda")) is supported

    def test_use_bf16_cpu(self, monkeypatch):
        """
        FUNCTION NAME: test_use_bf16_cpu

        DESCRIPTION: On the CPU, bf16 is never used, and the
        GPU support check is not even called (it can fail on
        machines without CUDA).

        INPUTS: monkeypatch (pytest fixture)

        OUTPUTS: None
        """
        def fail(*args, **kwargs):
            raise AssertionError("is_bf16_supported called for a CPU device")

        monkeypatch.setattr(torch.cuda, "is_bf16_supported", fail)
        assert train.use_bf16(torch.device("cpu")) is False

    def test_train_one_epoch_updates_and_steps(self, tmp_path):
        """
        FUNCTION NAME: test_train_one_epoch_updates_and_steps

        DESCRIPTION: One epoch changes the weights, steps the
        scheduler once per batch, returns a finite float, and
        leaves the model in train mode.

        INPUTS: tmp_path (pytest fixture)

        OUTPUTS: None
        """
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 3)
        before = model.weight.detach().clone()
        loader = make_loader(torch.randn(10, 4), torch.randint(0, 2, (10, 3)).float(), 4)
        optimizer, scheduler = train.build_optimizer(
            model, make_config(tmp_path, epochs=2), steps_per_epoch=len(loader))

        loss = train.train_one_epoch(model, loader, torch.nn.BCEWithLogitsLoss(),
                                     optimizer, scheduler, torch.device("cpu"), False)

        assert isinstance(loss, float)
        assert loss == loss and loss > 0
        assert not torch.equal(model.weight, before)
        assert scheduler.last_epoch == len(loader)
        assert model.training

    def test_train_one_epoch_loss_falls(self, tmp_path):
        """
        FUNCTION NAME: test_train_one_epoch_loss_falls

        DESCRIPTION: On an easy problem, the mean training loss
        goes down over several epochs.

        INPUTS: tmp_path (pytest fixture)

        OUTPUTS: None
        """
        torch.manual_seed(0)
        inputs = torch.randn(32, 4)
        labels = (inputs[:, :2] > 0).float()
        model = torch.nn.Linear(4, 2)
        loader = make_loader(inputs, labels, 8)
        optimizer, scheduler = train.build_optimizer(
            model, make_config(tmp_path, epochs=20, lr=0.05), steps_per_epoch=len(loader))
        loss_fn = torch.nn.BCEWithLogitsLoss()

        losses = [train.train_one_epoch(model, loader, loss_fn, optimizer, scheduler,
                                        torch.device("cpu"), False)
                  for _ in range(20)]
        assert losses[-1] < losses[0] / 2

    def test_evaluate_exact_values(self):
        """
        FUNCTION NAME: test_evaluate_exact_values

        DESCRIPTION: With all-zero logits the BCE loss is
        log(2), and probability 0.5 counts as a positive, so
        accuracy is the fraction of labels that are 1.

        INPUTS: None

        OUTPUTS: None
        """
        labels = torch.tensor([[1., 0.], [1., 1.], [0., 0.], [1., 0.], [0., 1.]])
        loader = make_loader(torch.randn(5, 4), labels, 2)
        model = ZeroModel(2)

        val_loss, val_accuracy = train.evaluate(model, loader, torch.nn.BCEWithLogitsLoss(),
                                                torch.device("cpu"), False)

        assert val_loss == pytest.approx(torch.log(torch.tensor(2.0)).item())
        assert val_accuracy == pytest.approx(5 / 10)
        assert not model.training

    def test_evaluate_weights_short_batch(self):
        """
        FUNCTION NAME: test_evaluate_weights_short_batch

        DESCRIPTION: The val loss from uneven batches (2, 2, 1)
        equals the loss over the whole set at once, and
        evaluate does not change the weights.

        INPUTS: None

        OUTPUTS: None
        """
        torch.manual_seed(0)
        inputs = torch.randn(5, 4)
        labels = torch.randint(0, 2, (5, 3)).float()
        model = torch.nn.Linear(4, 3)
        before = model.weight.detach().clone()
        loss_fn = torch.nn.BCEWithLogitsLoss()

        val_loss, _ = train.evaluate(model, make_loader(inputs, labels, 2), loss_fn,
                                     torch.device("cpu"), False)

        with torch.no_grad():
            expected = loss_fn(model(inputs), labels).item()
        assert val_loss == pytest.approx(expected, rel=1e-5)
        assert torch.equal(model.weight, before)

    def test_save_history(self, tmp_path):
        """
        FUNCTION NAME: test_save_history

        DESCRIPTION: The history is written as a JSON list of
        dicts (one per epoch), creating missing folders.

        INPUTS: tmp_path (pytest fixture)

        OUTPUTS: None
        """
        history = [train.EpochStats(1, 0.5, 0.4, 0.8, 1e-3, seconds=2.0),
                   train.EpochStats(2, 0.3, 0.35, 0.9, 5e-4, seconds=2.5)]
        path = tmp_path / "nested" / "run_history.json"

        train.save_history(path, history)

        data = json.loads(path.read_text())
        assert data[0] == {"epoch": 1, "train_loss": 0.5, "val_loss": 0.4,
                           "val_accuracy": 0.8, "lr": 1e-3, "seconds": 2.0}
        assert [train.EpochStats(**row) for row in data] == history

    def test_history_path_for(self):
        """
        FUNCTION NAME: test_history_path_for

        DESCRIPTION: The history file sits next to the
        checkpoint and is named <stem>_history.json.

        INPUTS: None

        OUTPUTS: None
        """
        path = train.history_path_for("checkpoints/model.pt")
        assert path.as_posix() == "checkpoints/model_history.json"

    def test_checkpoint_round_trip(self, tmp_path):
        """
        FUNCTION NAME: test_checkpoint_round_trip

        DESCRIPTION: A saved checkpoint reloads with the same
        weights, classes, thresholds, and extra keys, in eval
        mode.

        INPUTS: tmp_path (pytest fixture)

        OUTPUTS: None
        """
        torch.manual_seed(0)
        classes = ["bus", "bicycle", "hydrant"]
        original = model_lib.create_model(num_classes=3, pretrained=False)
        path = tmp_path / "sub" / "ckpt.pt"

        model_lib.save_checkpoint(path, original, classes,
                                  thresholds={"bus": 0.7},
                                  extra={"best_epoch": 3, "history": [{"epoch": 1}]})
        loaded, checkpoint = model_lib.load_checkpoint(path, expected_classes=classes)

        assert checkpoint["arch"] == model_lib.MODEL_ARCH
        assert checkpoint["classes"] == classes
        assert checkpoint["thresholds"] == {"bus": 0.7}
        assert checkpoint["best_epoch"] == 3
        assert checkpoint["history"] == [{"epoch": 1}]
        assert not loaded.training
        for name, tensor in original.state_dict().items():
            assert torch.equal(loaded.state_dict()[name], tensor)

    def test_save_checkpoint_rejects_bad_input(self, tmp_path):
        """
        FUNCTION NAME: test_save_checkpoint_rejects_bad_input

        DESCRIPTION: save_checkpoint raises ValueError for a
        head/class-count mismatch, a threshold on an unknown
        class, and extra keys that clash with the core keys.

        INPUTS: tmp_path (pytest fixture)

        OUTPUTS: None
        """
        net = model_lib.create_model(num_classes=2, pretrained=False)
        path = tmp_path / "ckpt.pt"

        with pytest.raises(ValueError):
            model_lib.save_checkpoint(path, net, ["a", "b", "c"])
        with pytest.raises(ValueError):
            model_lib.save_checkpoint(path, net, ["a", "b"], thresholds={"z": 0.5})
        with pytest.raises(ValueError):
            model_lib.save_checkpoint(path, net, ["a", "b"], extra={"classes": ["x"]})
        assert not path.exists()

    def test_load_checkpoint_rejects_mismatch(self, tmp_path):
        """
        FUNCTION NAME: test_load_checkpoint_rejects_mismatch

        DESCRIPTION: load_checkpoint raises ValueError when the
        expected classes differ in content or order, or the
        arch is wrong.

        INPUTS: tmp_path (pytest fixture)

        OUTPUTS: None
        """
        net = model_lib.create_model(num_classes=2, pretrained=False)
        path = tmp_path / "ckpt.pt"
        model_lib.save_checkpoint(path, net, ["bus", "hydrant"])

        with pytest.raises(ValueError):
            model_lib.load_checkpoint(path, expected_classes=["hydrant", "bus"])
        with pytest.raises(ValueError):
            model_lib.load_checkpoint(path, expected_classes=["bus", "bicycle"])

        checkpoint = torch.load(path, weights_only=True)
        checkpoint["arch"] = "resnet50"
        torch.save(checkpoint, path)
        with pytest.raises(ValueError):
            model_lib.load_checkpoint(path)

    def test_fit_synthetic(self, tmp_path):
        """
        FUNCTION NAME: test_fit_synthetic

        DESCRIPTION: A tiny two-epoch synthetic run returns one
        EpochStats per epoch and writes a loadable checkpoint
        (with config, best epoch, and history) plus the
        history JSON.

        INPUTS: tmp_path (pytest fixture)

        OUTPUTS: None
        """
        config = make_config(tmp_path, epochs=2)

        history = train.fit(config, verbose=False)

        assert [stats.epoch for stats in history] == [1, 2]
        assert all(0.0 <= stats.val_accuracy <= 1.0 for stats in history)
        assert history[0].lr == pytest.approx(config.lr)

        _, checkpoint = model_lib.load_checkpoint(
            config.out_path, expected_classes=["bus", "bicycle", "traffic_light", "hydrant"])
        best = min(history, key=lambda stats: stats.val_loss)
        assert checkpoint["best_epoch"] == best.epoch
        assert checkpoint["config"]["epochs"] == 2
        assert len(checkpoint["history"]) == 2

        saved = json.loads((tmp_path / "model_history.json").read_text())
        assert [train.EpochStats(**row) for row in saved] == history

    def test_fit_is_reproducible(self, tmp_path):
        """
        FUNCTION NAME: test_fit_is_reproducible

        DESCRIPTION: Two runs with the same seed give the same
        history (timings are ignored by EpochStats ==).

        INPUTS: tmp_path (pytest fixture)

        OUTPUTS: None
        """
        first = train.fit(make_config(tmp_path, out_path=str(tmp_path / "a.pt")), verbose=False)
        second = train.fit(make_config(tmp_path, out_path=str(tmp_path / "b.pt")), verbose=False)
        assert first == second

    def test_parse_args_defaults(self):
        """
        FUNCTION NAME: test_parse_args_defaults

        DESCRIPTION: No flags gives the default TrainConfig.

        INPUTS: None

        OUTPUTS: None
        """
        assert train.parse_args([]) == train.TrainConfig()

    def test_parse_args_flags(self):
        """
        FUNCTION NAME: test_parse_args_flags

        DESCRIPTION: Every flag maps onto its TrainConfig field.

        INPUTS: None

        OUTPUTS: None
        """
        config = train.parse_args([
            "--data", "My-Data", "--epochs", "3", "--batch-size", "64",
            "--lr", "0.001", "--weight-decay", "0", "--device", "cpu",
            "--num-workers", "2", "--seed", "7", "--out", "runs/x.pt",
            "--no-pretrained"])

        assert config == train.TrainConfig(
            data_dir="My-Data", synthetic=False, epochs=3, batch_size=64,
            lr=0.001, weight_decay=0.0, device="cpu", num_workers=2, seed=7,
            out_path="runs/x.pt", pretrained=False)
        assert train.parse_args(["--synthetic"]).synthetic is True

    @pytest.mark.parametrize("argv", [
        ["--data", "Image-Data", "--synthetic"],
        ["--device", "tpu"],
    ])
    def test_parse_args_rejects(self, argv):
        """
        FUNCTION NAME: test_parse_args_rejects

        DESCRIPTION: --data with --synthetic, or an unknown
        device, exits with an argparse error.

        INPUTS: argv (list of str)

        OUTPUTS: None
        """
        with pytest.raises(SystemExit):
            train.parse_args(argv)

    def test_main_out_of_memory(self, monkeypatch, capsys):
        """
        FUNCTION NAME: test_main_out_of_memory

        DESCRIPTION: A CUDA out-of-memory error makes main
        print a --batch-size 64 hint and return 1.

        INPUTS: monkeypatch, capsys (pytest fixtures)

        OUTPUTS: None
        """
        def out_of_memory(config, verbose=True):
            raise torch.cuda.OutOfMemoryError("fake")

        monkeypatch.setattr(train, "fit", out_of_memory)
        assert train.main(["--synthetic"]) == 1
        assert "--batch-size 64" in capsys.readouterr().err

    def test_main_missing_data(self, tmp_path, capsys):
        """
        FUNCTION NAME: test_main_missing_data

        DESCRIPTION: A data folder without train/ and val/
        makes main print an error and return 1.

        INPUTS: tmp_path (pytest fixture), capsys (pytest fixture)

        OUTPUTS: None
        """
        assert train.main(["--data", str(tmp_path), "--device", "cpu"]) == 1
        assert "does not exist" in capsys.readouterr().err

    def test_main_success(self, monkeypatch):
        """
        FUNCTION NAME: test_main_success

        DESCRIPTION: main returns 0 and passes the parsed
        config to fit.

        INPUTS: monkeypatch (pytest fixture)

        OUTPUTS: None
        """
        seen = []
        monkeypatch.setattr(train, "fit",
                            lambda config, verbose=True: seen.append(config) or [])
        assert train.main(["--synthetic", "--epochs", "2"]) == 0
        assert seen[0].synthetic is True and seen[0].epochs == 2
