#############################################################
#   FILE NAME: test_model_solver.py
#
#   DESCRIPTION: This file contains unit tests for
#   FolderTileSource (Library/tile_sources.py) and
#   ModelSolver (Library/solvers.py). They live outside
#   unit_test.py so they run on machines without tkinter.
#   Models are built with pretrained=False and given fixed
#   head biases, so predictions are known exactly.
#
#   Authors: Group 2
#############################################################

#############################################################
#   Libraries
#############################################################

import random

import pytest
import torch
from PIL import Image

#############################################################
#   Files Import
#############################################################

from Library.benchmark import compare_solvers
from Library.captcha_env import CaptchaEnv, Challenge, NUM_TILES
from Library.model import create_model, save_checkpoint
from Library.solvers import ModelSolver, OracleSolver, make_solver
from Library.tile_sources import FolderTileSource, SyntheticTileSource, TILE_SIZE

#############################################################
#   Global Constants
#############################################################

CLASSES = ["bicycle", "bus", "hydrant", "traffic_light"]
COLORS = {"bicycle": (200, 0, 0), "bus": (0, 200, 0),
          "hydrant": (0, 0, 200), "traffic_light": (200, 200, 0)}

#############################################################
#   FUNCTIONS
#############################################################

def make_split(root, per_class=3, classes=CLASSES):
    """
    FUNCTION NAME: make_split

    DESCRIPTION: Writes a tiny split folder of solid-color
    images, root/<class>/<class>_<i>.png, with odd sizes so
    resizing is exercised.

    INPUTS: root (Path), per_class (int), classes (list of str)

    OUTPUTS: root (Path)
    """
    for label in classes:
        folder = root / label
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(per_class):
            Image.new("RGB", (40 + index, 30), COLORS.get(label, (9, 9, 9))).save(
                folder / f"{label}_{index}.png")
    return root


def make_checkpoint(path, biases, classes=CLASSES, thresholds=None):
    """
    FUNCTION NAME: make_checkpoint

    DESCRIPTION: Saves a ResNet-18 whose head ignores the
    image (zero weights) and outputs the given logit per
    class, so every tile gets probability sigmoid(bias).

    INPUTS: path (Path), biases (list of float, one per
    class), classes (list of str), thresholds (dict or None)

    OUTPUTS: path (Path)
    """
    model = create_model(num_classes=len(classes), pretrained=False)
    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.bias.copy_(torch.tensor(biases, dtype=torch.float32))
    save_checkpoint(path, model, classes, thresholds=thresholds)
    return path


def make_challenge(target, tiles):
    """
    FUNCTION NAME: make_challenge

    DESCRIPTION: Builds a challenge whose labels are junk, to
    prove ModelSolver never reads them.

    INPUTS: target (str), tiles (list of PIL images)

    OUTPUTS: Challenge
    """
    return Challenge(id="test", target=target, tiles=tiles, labels=["junk"] * len(tiles))

#############################################################
#   CLASSES
#############################################################

class TestFolderTileSource:
    """
    CLASS NAME: TestFolderTileSource

    DESCRIPTION: This class contains unit tests for
    FolderTileSource.
    """

    def test_init_reads_class_folders(self, tmp_path):
        """
        FUNCTION NAME: test_init_reads_class_folders

        DESCRIPTION: Classes default to the sorted subfolder
        names, every image is indexed, and other files are
        ignored.

        INPUTS: tmp_path (pytest fixture)

        OUTPUTS: None
        """
        make_split(tmp_path, per_class=3)
        (tmp_path / "bus" / "notes.txt").write_text("not an image")

        source = FolderTileSource(tmp_path)

        assert source.classes == CLASSES
        assert all(len(source.files[label]) == 3 for label in CLASSES)

    def test_init_explicit_classes(self, tmp_path):
        """
        FUNCTION NAME: test_init_explicit_classes

        DESCRIPTION: An explicit class list is kept in its
        given order and limits the classes used.

        INPUTS: tmp_path (pytest fixture)

        OUTPUTS: None
        """
        make_split(tmp_path)
        source = FolderTileSource(tmp_path, classes=["bus", "bicycle"])
        assert source.classes == ["bus", "bicycle"]
        assert set(source.files) == {"bus", "bicycle"}

    def test_init_errors(self, tmp_path):
        """
        FUNCTION NAME: test_init_errors

        DESCRIPTION: A missing folder, a missing class folder,
        an empty class folder, fewer than 2 classes, and a bad
        size all raise.

        INPUTS: tmp_path (pytest fixture)

        OUTPUTS: None
        """
        with pytest.raises(FileNotFoundError):
            FolderTileSource(tmp_path / "missing")

        make_split(tmp_path / "split")
        with pytest.raises(FileNotFoundError):
            FolderTileSource(tmp_path / "split", classes=["bus", "car"])
        with pytest.raises(ValueError):
            FolderTileSource(tmp_path / "split", classes=["bus"])
        with pytest.raises(ValueError):
            FolderTileSource(tmp_path / "split", size=0)

        (tmp_path / "split" / "empty").mkdir()
        with pytest.raises(ValueError, match="no images"):
            FolderTileSource(tmp_path / "split")

    def test_get_tile(self, tmp_path):
        """
        FUNCTION NAME: test_get_tile

        DESCRIPTION: A tile is an RGB image of the source's
        size, comes from the right class folder, and records
        its file in info["source"].

        INPUTS: tmp_path (pytest fixture)

        OUTPUTS: None
        """
        make_split(tmp_path)
        source = FolderTileSource(tmp_path, size=64)

        tile = source.get_tile("hydrant", random.Random(0))

        assert tile.mode == "RGB"
        assert tile.size == (64, 64)
        assert tile.getpixel((10, 10)) == COLORS["hydrant"]
        assert "hydrant" in tile.info["source"]

        with pytest.raises(ValueError):
            source.get_tile("car", random.Random(0))
        with pytest.raises(TypeError):
            source.get_tile("bus", 0)

    def test_get_tile_uses_only_rng(self, tmp_path):
        """
        FUNCTION NAME: test_get_tile_uses_only_rng

        DESCRIPTION: The same rng seed picks the same files,
        so the source keeps no random state of its own.

        INPUTS: tmp_path (pytest fixture)

        OUTPUTS: None
        """
        make_split(tmp_path, per_class=5)
        source = FolderTileSource(tmp_path)

        def picks(seed):
            rng = random.Random(seed)
            return [source.get_tile("bus", rng).info["source"] for _ in range(10)]

        assert picks(3) == picks(3)
        assert len(set(picks(3))) > 1

    def test_env_fills_sources(self, tmp_path):
        """
        FUNCTION NAME: test_env_fills_sources

        DESCRIPTION: Challenges from a folder source list each
        tile's file, in the right class folder; synthetic
        challenges leave sources empty.

        INPUTS: tmp_path (pytest fixture)

        OUTPUTS: None
        """
        make_split(tmp_path)
        challenge = CaptchaEnv(FolderTileSource(tmp_path), seed=1).new_challenge()

        assert len(challenge.sources) == NUM_TILES
        assert all(f"{label}_" in source for label, source in zip(challenge.labels, challenge.sources))
        assert all(tile.size == (TILE_SIZE, TILE_SIZE) for tile in challenge.tiles)

        assert CaptchaEnv(SyntheticTileSource(), seed=1).new_challenge().sources == []

    def test_benchmark_with_folder_source(self, tmp_path):
        """
        FUNCTION NAME: test_benchmark_with_folder_source

        DESCRIPTION: One folder source can be shared across
        compare_solvers runs, and the oracle passes every
        challenge.

        INPUTS: tmp_path (pytest fixture)

        OUTPUTS: None
        """
        make_split(tmp_path)
        source = FolderTileSource(tmp_path)

        results = compare_solvers(source, [OracleSolver(), OracleSolver()], n=20, seed=5)

        assert results[0].pass_rate == 1.0
        assert results[0] == results[1]


class TestModelSolver:
    """
    CLASS NAME: TestModelSolver

    DESCRIPTION: This class contains unit tests for
    ModelSolver and make_solver("model").
    """

    def test_selects_by_target_probability(self, tmp_path):
        """
        FUNCTION NAME: test_selects_by_target_probability

        DESCRIPTION: With a confident "bus" logit the solver
        picks every tile for a bus challenge and none for a
        hydrant challenge, without reading the labels.

        INPUTS: tmp_path (pytest fixture)

        OUTPUTS: None
        """
        path = make_checkpoint(tmp_path / "m.pt", [-5.0, 5.0, -5.0, -5.0])
        solver = ModelSolver(path)
        tiles = [SyntheticTileSource().get_tile("bus", random.Random(i)) for i in range(NUM_TILES)]

        assert solver.name == "model"
        assert solver.solve(make_challenge("bus", tiles)) == list(range(NUM_TILES))
        assert solver.solve(make_challenge("hydrant", tiles)) == []

    def test_thresholds(self, tmp_path):
        """
        FUNCTION NAME: test_thresholds

        DESCRIPTION: A probability of exactly 0.5 passes the
        default threshold; a stored checkpoint threshold
        raises the bar, and an explicit one overrides it.

        INPUTS: tmp_path (pytest fixture)

        OUTPUTS: None
        """
        tiles = [SyntheticTileSource().get_tile("bus", random.Random(i)) for i in range(NUM_TILES)]
        challenge = make_challenge("bus", tiles)

        plain = make_checkpoint(tmp_path / "plain.pt", [0.0] * 4)
        assert ModelSolver(plain).solve(challenge) == list(range(NUM_TILES))

        tuned = make_checkpoint(tmp_path / "tuned.pt", [0.0] * 4, thresholds={"bus": 0.6})
        assert ModelSolver(tuned).thresholds == {"bicycle": 0.5, "bus": 0.6,
                                                 "hydrant": 0.5, "traffic_light": 0.5}
        assert ModelSolver(tuned).solve(challenge) == []
        assert ModelSolver(tuned, thresholds={"bus": 0.4}).solve(challenge) == list(range(NUM_TILES))

        with pytest.raises(ValueError):
            ModelSolver(plain, thresholds={"car": 0.5})

    def test_tile_probabilities(self, tmp_path):
        """
        FUNCTION NAME: test_tile_probabilities

        DESCRIPTION: tile_probabilities returns one sigmoid
        probability per class per tile, in checkpoint class
        order.

        INPUTS: tmp_path (pytest fixture)

        OUTPUTS: None
        """
        biases = [-1.0, 0.0, 1.0, 2.0]
        solver = ModelSolver(make_checkpoint(tmp_path / "m.pt", biases))
        tiles = [SyntheticTileSource().get_tile("bus", random.Random(i)) for i in range(3)]

        probabilities = solver.tile_probabilities(tiles)

        assert probabilities.shape == (3, 4)
        expected = torch.sigmoid(torch.tensor(biases)).expand(3, 4)
        assert torch.allclose(probabilities, expected, atol=1e-5)

    def test_class_checks(self, tmp_path):
        """
        FUNCTION NAME: test_class_checks

        DESCRIPTION: The solver rejects a source whose classes
        the checkpoint lacks, accepts one in a different
        order, and rejects a challenge target it doesn't know.

        INPUTS: tmp_path (pytest fixture)

        OUTPUTS: None
        """
        path = make_checkpoint(tmp_path / "m.pt", [0.0] * 4)

        ModelSolver(path, required_classes=["bus", "bicycle", "traffic_light", "hydrant"])
        with pytest.raises(ValueError, match="fire_hydrant"):
            ModelSolver(path, required_classes=["bus", "fire_hydrant"])

        tiles = [SyntheticTileSource().get_tile("bus", random.Random(0))] * NUM_TILES
        with pytest.raises(ValueError):
            ModelSolver(path).solve(make_challenge("car", tiles))

    def test_make_solver_model(self, tmp_path):
        """
        FUNCTION NAME: test_make_solver_model

        DESCRIPTION: make_solver builds a ModelSolver from a
        checkpoint and requires one.

        INPUTS: tmp_path (pytest fixture)

        OUTPUTS: None
        """
        path = make_checkpoint(tmp_path / "m.pt", [0.0] * 4)
        assert isinstance(make_solver("model", checkpoint=path), ModelSolver)
        with pytest.raises(ValueError):
            make_solver("model")

    def test_benchmark_runs(self, tmp_path):
        """
        FUNCTION NAME: test_benchmark_runs

        DESCRIPTION: A model that selects every tile runs
        through compare_solvers on a folder source: recall is
        1, and it never passes since some tiles are decoys.

        INPUTS: tmp_path (pytest fixture)

        OUTPUTS: None
        """
        make_split(tmp_path / "split")
        source = FolderTileSource(tmp_path / "split")
        solver = ModelSolver(make_checkpoint(tmp_path / "m.pt", [5.0] * 4),
                             required_classes=source.classes)

        result, = compare_solvers(source, [solver], n=5, seed=0)

        assert result.solver_name == "model"
        assert result.recall == 1.0
        assert result.passes == 0
