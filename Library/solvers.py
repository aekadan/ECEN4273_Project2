#############################################################
#   FILE NAME: solvers.py
#
#   DESCRIPTION: This is the solvers file for the project.
#   It contains the implementation of various solvers used
#   in the project.
#
#   Authors: Group 2
#############################################################

#############################################################
#   Libraries
#############################################################
import random

import torch

#############################################################
#   Files Import
#############################################################
from .captcha_env import NUM_TILES
from .dataset import get_transform
from .model import load_checkpoint, predict

#############################################################
#   Global Constants
#############################################################
PROBABILITY = 0.5
DEFAULT_THRESHOLD = 0.5

#############################################################
#   CLASSES
#############################################################
class Solver:
    """
    CLASS NAME: Solver

    DESCRIPTION: This is the base class for all solvers in the project.

    INPUTS: name, seed

    OUTPUTS: None
    """
    def __init__(self, name, seed=None):
        """
        FUNCTION NAME: __init__

        DESCRIPTION: This stores the solver's name and random generator.

        INPUTS: name (str), seed (int or None)

        OUTPUTS: None
        """
        self.name = name
        self.rng = random.Random(seed)

    def solve(self, challenge):
        """
        FUNCTION NAME: solve

        DESCRIPTION: This selects the tiles to submit for a challenge.

        INPUTS: challenge (Challenge)

        OUTPUTS: list of selected tile indices (ints 0-8)
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


class OracleSolver(Solver):
    """
    CLASS NAME: OracleSolver

    DESCRIPTION: This solver always returns the correct answer.

    INPUTS: None

    OUTPUTS: None
    """
    def __init__(self):
        """
        FUNCTION NAME: __init__

        DESCRIPTION: This sets the solver's name.

        INPUTS: None

        OUTPUTS: None
        """
        super().__init__("oracle")

    def solve(self, challenge):
        """
        FUNCTION NAME: solve

        DESCRIPTION: This returns the correct tiles for a challenge.

        INPUTS: challenge (Challenge)

        OUTPUTS: list of selected tile indices (ints 0-8)
        """
        return challenge.answer


class RandomSolver(Solver):
    """
    CLASS NAME: RandomSolver

    DESCRIPTION: This solver selects each tile on a coin flip.

    INPUTS: seed

    OUTPUTS: None
    """
    def __init__(self, seed=None):
        """
        FUNCTION NAME: __init__

        DESCRIPTION: This sets the solver's name and seed.

        INPUTS: seed (int or None)

        OUTPUTS: None
        """
        super().__init__("random", seed)

    def solve(self, challenge):
        """
        FUNCTION NAME: solve

        DESCRIPTION: This selects each tile with a 50% chance.

        INPUTS: challenge (Challenge)

        OUTPUTS: list of selected tile indices (ints 0-8)
        """
        selected = []
        for index in range(NUM_TILES):
            if self.rng.random() < PROBABILITY:
                selected.append(index)
        return selected


class NoisySolver(Solver):
    """
    CLASS NAME: NoisySolver

    DESCRIPTION: This solver gets each tile right with probability p.

    INPUTS: p, seed

    OUTPUTS: None
    """
    def __init__(self, p, seed=None):
        """
        FUNCTION NAME: __init__

        DESCRIPTION: This validates and stores p, then sets the name and seed.

        INPUTS: p (float from 0 to 1), seed (int or None)

        OUTPUTS: None
        """
        super().__init__("noisy", seed)
        if p < 0 or p > 1:
            raise ValueError("p must be between 0 and 1")
        self.p = p

    def solve(self, challenge):
        """
        FUNCTION NAME: solve

        DESCRIPTION: This starts from the answer and flips each tile with probability 1 - p.

        INPUTS: challenge (Challenge)

        OUTPUTS: list of selected tile indices (ints 0-8)
        """
        selected = []
        answer_set = set(challenge.answer)
        for index in range(NUM_TILES):
            if index in answer_set:
                if self.rng.random() < self.p:
                    selected.append(index)
            else:
                if not (self.rng.random() < self.p):
                    selected.append(index)
        return selected


class ModelSolver(Solver):
    """
    CLASS NAME: ModelSolver

    DESCRIPTION: This solver runs the 9 tiles through a trained
    checkpoint and selects every tile whose probability for the target
    class is at or above that class's threshold. It only reads
    challenge.tiles and challenge.target, never the answer or labels.

    INPUTS: checkpoint_path, device, required_classes, thresholds

    OUTPUTS: None
    """
    def __init__(self, checkpoint_path, device="cpu", required_classes=None, thresholds=None):
        """
        FUNCTION NAME: __init__

        DESCRIPTION: This loads the checkpoint, checks it knows every
        required class, and sets up the per-class thresholds (explicit
        thresholds, then the checkpoint's, then DEFAULT_THRESHOLD).

        INPUTS: checkpoint_path (str or Path), device (str or
        torch.device), required_classes (list of str the challenges can
        target, e.g. source.classes, or None), thresholds (dict of
        class -> float overriding the checkpoint's, or None)

        OUTPUTS: None
        """
        super().__init__("model")
        self.device = torch.device(device)
        self.model, self.checkpoint = load_checkpoint(checkpoint_path, device=self.device)
        self.classes = list(self.checkpoint["classes"])

        if required_classes is not None:
            missing = [label for label in required_classes if label not in self.classes]
            if missing:
                raise ValueError(f"Checkpoint classes {self.classes} are missing {missing}.")

        self.thresholds = {label: DEFAULT_THRESHOLD for label in self.classes}
        for override in (self.checkpoint.get("thresholds"), thresholds):
            if override:
                unknown = [label for label in override if label not in self.classes]
                if unknown:
                    raise ValueError(f"Thresholds name unknown classes: {unknown}")
                self.thresholds.update({label: float(value) for label, value in override.items()})

        self.transform = get_transform()

    def tile_probabilities(self, tiles):
        """
        FUNCTION NAME: tile_probabilities

        DESCRIPTION: This returns the model's probability for every class
        on every tile (useful for threshold tuning and inspecting mistakes).

        INPUTS: tiles (list of PIL images)

        OUTPUTS: N x C float tensor on the CPU, columns in self.classes order
        """
        batch = torch.stack([self.transform(tile.convert("RGB")) for tile in tiles])
        return predict(self.model, batch, self.device).float().cpu()

    def solve(self, challenge):
        """
        FUNCTION NAME: solve

        DESCRIPTION: This selects the tiles whose target-class probability
        is at or above the target's threshold.

        INPUTS: challenge (Challenge)

        OUTPUTS: sorted list of selected tile indices (ints 0-8)
        """
        if challenge.target not in self.classes:
            raise ValueError(f"Target {challenge.target!r} is not one of the model's classes {self.classes}.")
        column = self.classes.index(challenge.target)
        probabilities = self.tile_probabilities(challenge.tiles)[:, column].tolist()
        threshold = self.thresholds[challenge.target]
        return [index for index, probability in enumerate(probabilities) if probability >= threshold]


#############################################################
#   FUNCTIONS
#############################################################
def make_solver(name, seed=None, p=None, checkpoint=None, device="cpu", required_classes=None):
    """
    FUNCTION NAME: make_solver

    DESCRIPTION: This builds a solver from its name.

    INPUTS: name (str: "oracle", "random", "noisy", or "model"),
    seed (int or None), p (float, noisy only), checkpoint (path, model
    only), device (str or torch.device, model only), required_classes
    (list of str, model only)

    OUTPUTS: Solver
    """
    if name == "oracle":
        solve = OracleSolver()
    elif name == "random":
        solve = RandomSolver(seed)
    elif name == "noisy":
        if p is None:
            raise ValueError("p must be provided for NoisySolver")
        else:
            solve = NoisySolver(p, seed)
    elif name == "model":
        if checkpoint is None:
            raise ValueError("checkpoint must be provided for ModelSolver")
        solve = ModelSolver(checkpoint, device=device, required_classes=required_classes)
    else:
        raise ValueError(f"Unknown solver name: {name}")
    return solve
