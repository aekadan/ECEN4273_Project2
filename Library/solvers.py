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

#############################################################
#   Files Import
#############################################################
from .captcha_env import NUM_TILES

#############################################################
#   Global Constants
#############################################################
PROBABILITY = 0.5

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


#############################################################
#   FUNCTIONS
#############################################################
def make_solver(name, seed=None, p=None):
    """
    FUNCTION NAME: make_solver

    DESCRIPTION: This builds a solver from its name.

    INPUTS: name (str), seed (int or None), p (float, noisy only)

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
    else:
        raise ValueError(f"Unknown solver name: {name}")
    return solve
