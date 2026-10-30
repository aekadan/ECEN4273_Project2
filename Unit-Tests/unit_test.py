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
sys.path.insert(0, str(PROJECT_ROOT / "captcha-env"))

from captcha_env import CaptchaEnv, Challenge, NUM_TILES
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