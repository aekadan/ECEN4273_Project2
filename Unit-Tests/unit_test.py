#############################################################
#   FILE NAME: unit-tests.py
#
#   DESCRIPTION: This is the unit tests file for the project.
#
#   Authors: Group 2
#############################################################

#############################################################
#   Libraries
#############################################################

import sys
from pathlib import Path

#############################################################
#   Files Import
#############################################################

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Library"))

from captcha_env import CaptchaEnv, Challenge, GradeResult, NUM_TILES
from tile_sources import SyntheticTileSource, DEFAULT_CLASSES
from main import main

#############################################################
#   FUNCTIONS
#############################################################
class TestMain:
    """
    CLASS NAME: TestMain

    DESCRIPTION: This class contains unit tests for the main function.
    """
    def test_main(self):
        """
        FUNCTION: test_main

        DESCRIPTION: This is the unit test for the main function.

        INPUTS: None

        Outputs: None
        """
        assert main() == 0

class TestCaptchaEnv:
    """
    CLASS NAME: TestCaptchaEnv

    DESCRIPTION: This class contains unit tests for the captcha environment.
    """
    def test_new_challenge(self):
        """
        FUNCTION: test_new_challenge

        DESCRIPTION: This is the unit test for the new_challenge function.

        INPUTS: None

        Outputs: None
        """
        # Create a mock source with a get_tile method
        class MockSource:
            def __init__(self):
                self.classes = ['cat', 'dog', 'bus']
            def get_tile(self, label, rng):
                return f"Tile for {label}"

        # Create an instance of the captcha environment with the mock source
        env = CaptchaEnv(source=MockSource(), min_targets=1, max_targets=3)
        
        # Generate a new challenge
        challenge = env.new_challenge()
        
        # Check that the challenge has the correct attributes
        assert isinstance(challenge, Challenge)
        assert challenge.target in challenge.labels
        assert len(challenge.tiles) == NUM_TILES

    def test_grade(self):
        """
        FUNCTION: test_grade

        DESCRIPTION: This is the unit test for the grade function.

        INPUTS: None

        Outputs: None
        """
        class MockSource:
            classes = ["cat", "dog"]
            def get_tile(self, label, rng):
                return label

        env = CaptchaEnv(source=MockSource())

        # Known challenge: tiles 0, 4 and 8 are cats
        labels = ["cat", "dog", "dog",
                "dog", "cat", "dog",
                "dog", "dog", "cat"]
        challenge = Challenge(id="test", target="cat", tiles=labels, labels=labels)

        # Correct selection passes
        result = env.grade(challenge, [0, 4, 8])
        assert result.passed
        assert result.true_positives == 3
        assert result.true_negatives == 6

        # Missing a tile fails
        result = env.grade(challenge, [0, 4])
        assert not result.passed
        assert result.false_negatives == 1

        # Selecting an extra tile fails
        result = env.grade(challenge, [0, 4, 8, 1])
        assert not result.passed
        assert result.false_positives == 1

class TestTileSources:
    """
    CLASS NAME: TestTileSources

    DESCRIPTION: This class contains unit tests for the tile sources.
    """