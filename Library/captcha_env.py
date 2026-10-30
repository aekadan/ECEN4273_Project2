#############################################################
#   FILE NAME: captcha_env.py
#
#   DESCRIPTION: This is the captcha environment file in which 
#   an offline captcha environment is created for the agent 
#   to interact with.
#
#   Authors: Group 2
#############################################################

#############################################################
#   Libraries
#############################################################

import random
import uuid
import numbers

from dataclasses import dataclass, field
from PIL import Image, ImageDraw
#############################################################
#   Files Import
#############################################################

#############################################################
#   Global Constants
#############################################################

GRID_ROWS = 3
GRID_COLS = 3
NUM_TILES = GRID_ROWS * GRID_COLS

PLURALS = {"bus": "busses"}

#############################################################
#   CLASSES
#############################################################
@dataclass
class Challenge:
    """
    CLASS NAME: Challenge
    
    DESCRIPTION: This class represents a single 3x3 captcha challenge.
    The tiles are labeled from 0 to 8, starting from the top-left corner 
    and moving row-wise to the bottom-right corner.

    INPUTS: id, target, tiles, labels, sources

    OUTPUTS: None
    """
    id:str
    target:str
    tiles:list
    labels:list
    sources:list = field(default_factory=list)

    @property
    def answer(self):
        """
        FUNCTION NAME: answer

        DESCRIPTION: This function returns the indices of the tiles that match the target label.

        INPUTS: None

        OUTPUTS: answer
        """
        return [i for i, label in enumerate(self.labels) if label == self.target]

    @property
    def prompt(self):
        """
        FUNCTION NAME: prompt

        DESCRIPTION: This function generates a prompt for the captcha challenge.

        INPUTS: None

        OUTPUTS: prompt
        """
        name = self.target.replace("_", " ")
        plural = PLURALS.get(name, name if name.endswith("s") else name + "s")
        return f"Select all images with {plural}."

@dataclass
class GradeResult:
    """
    CLASS NAME: GradeResult

    DESCRIPTION: This class represents the result of grading a captcha challenge.

    INPUTS: passed, true_positives, false_positives, false_negatives, true_negatives

    OUTPUTS: None
    """
    # GradeResult declaration with attributes for grading results
    passed:bool
    true_positives:int
    false_positives:int
    false_negatives:int
    true_negatives:int

    @property
    def tiles_correct(self):
        """
        FUNCTION NAME: tiles_correct

        DESCRIPTION: This function returns the number of tiles that were correctly classified.

        INPUTS: None

        OUTPUTS: tiles_correct
        """
        return self.true_positives + self.true_negatives

class CaptchaEnv:
    """
    CLASS NAME: CaptchaEnv

    DESCRIPTION: This class represents the captcha environment in which the agent interacts with the challenges.

    INPUTS: source, min_targets, max_targets, seed

    OUTPUTS: None
    """
    def __init__(self, source, min_targets=2, max_targets=5, seed=None):
        if not 1 <= min_targets <= max_targets <= NUM_TILES:
            raise ValueError(f"min_targets and max_targets must be between 1 and {NUM_TILES}, and min_targets must be less than or equal to max_targets.")
        if source.classes is None or len(source.classes) < 2:
            raise ValueError("The source must have at least two classes.")
        # Initialize the CaptchaEnv with the provided parameters
        self.source = source
        self.min_targets = min_targets
        self.max_targets = max_targets
        self.seed = seed
        self.rng = random.Random(seed)

    def new_challenge(self):
        """
        FUNCTION NAME: new_challenge

        DESCRIPTION: This function generates a new captcha challenge.

        INPUTS: None

        OUTPUTS: challenge
        """
        classes = self.source.classes
        target = self.rng.choice(classes)
        decoys:list = [cls for cls in classes if cls != target]
        target_count:int = self.rng.randint(self.min_targets, self.max_targets)
        labels:list = [target] * target_count + self.rng.choices(decoys, k=NUM_TILES - target_count)
        # Shuffle the labels to randomize their positions
        self.rng.shuffle(labels)
        # Get the tiles based on the labels from the source
        tiles:list = [self.source.get_tile(label, self.rng) for label in labels]
        # Return a new Challenge object with a unique ID, target label, tiles, and labels
        return Challenge(id=str(uuid.uuid4()), target=target, tiles=tiles, labels=labels)

    def grade(self, challenge: Challenge, selection: list):

        """
        FUNCTION NAME: grade
        
        Description: This function grades the agent's selection for a given captcha challenge.
        
        INPUTS: challenge, selection (list of ints)
        
        OUTPUTS: GradeResult
        """
        if not isinstance(selection, list):
            raise TypeError("Selection must be a list of integers.")

        if not all(isinstance(s, numbers.Integral) for s in selection):
            raise TypeError("All elements in the selection must be integers.")

        if any(s >= NUM_TILES or s < 0 for s in selection):
            raise ValueError(f"Selection must be between 0 and {NUM_TILES - 1}.")
        
        if len(set(selection)) != len(selection):
            raise ValueError("Selection must not contain duplicate indices.")
        # Convert the challenge answer and selection to sets for easier comparison
        answer = set(challenge.answer)
        selection_set = set(selection)
        # Calculate true positives, false positives, false negatives, and true negatives
        true_positives = len(selection_set.intersection(answer))
        false_positives = len(selection_set.difference(answer))
        false_negatives = len(answer.difference(selection_set))
        true_negatives = NUM_TILES - (true_positives + false_positives + false_negatives)
        return GradeResult(
            true_positives=true_positives, false_positives=false_positives, 
            false_negatives=false_negatives, true_negatives=true_negatives, 
            passed=(false_positives == 0 and false_negatives == 0)
            )

#############################################################
#   FUNCTIONS
#############################################################
def render_grid(challenge: Challenge, tile_size=100):
    """
    FUNCTION NAME: render_grid

    DESCRIPTION: This function renders the captcha challenge grid as an image.

    INPUTS: challenge, tile_size

    OUTPUTS: img
    """
    