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
import random
import PIL

#############################################################
#   Files Import
#############################################################
from Library.gui import CaptchaApp
from Library.captcha_env import CaptchaEnv, Challenge, GradeResult, NUM_TILES
from Library.tile_sources import SyntheticTileSource, DEFAULT_CLASSES, TILE_SIZE
from main import main

#############################################################
#   TEST CLASSES
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
    def test_synthetic_tile_init(self):
        """
        FUNCTION: test_synthetic_tile_init

        DESCRIPTION: This is the unit test for the SyntheticTileSource class initialization.

        INPUTS: None

        Outputs: None
        """
        # Test with valid classes and size
        source = SyntheticTileSource(classes=["cat", "dog"], size=64)
        assert source.classes == ["cat", "dog"]
        assert source.size == 64

        # Test with default classes and size
        source = SyntheticTileSource()
        assert source.classes == DEFAULT_CLASSES
        assert source.size == 100

        # Test with invalid classes (less than 2)
        try:
            source = SyntheticTileSource(classes=["cat"])
            assert False, "Expected ValueError for less than 2 classes"
        except ValueError:
            pass

        # Test with duplicate classes
        try:
            source = SyntheticTileSource(classes=["cat", "cat"])
            assert False, "Expected ValueError for duplicate classes"
        except ValueError:
            pass

        # Test with invalid size (non-integer)
        try:
            source = SyntheticTileSource(size="64")
            assert False, "Expected TypeError for non-integer size"
        except TypeError:
            pass

        # Test with invalid size (negative integer)
        try:
            source = SyntheticTileSource(size=-64)
            assert False, "Expected ValueError for negative size"
        except ValueError:
            pass

    def test_synthetic_tile_get_tile(self):
        """
        FUNCTION: test_synthetic_tile_get_tile

        DESCRIPTION: This is the unit test for the get_tile method of the SyntheticTileSource class.

        INPUTS: None

        Outputs: None
        """
        source = SyntheticTileSource(classes=["cat", "dog"], size=64)
        rng = random.Random(42)

        # Test getting a tile for a valid label
        tile = source.get_tile("cat", rng)
        assert isinstance(tile, PIL.Image.Image)  # Assuming get_tile returns a PIL Image object

        # Test getting a tile for an invalid label
        try:
            tile = source.get_tile("bird", rng)
            assert False, "Expected ValueError for invalid label"
        except ValueError:
            pass

class TestCaptchaApp:
    """
    CLASS NAME: TestCaptchaApp

    DESCRIPTION: This class contains unit tests for the CaptchaApp class.
    """
    def test_init(self):
        """
        FUNCTION: test_init

        DESCRIPTION: This is the unit test for the __init__ method of the CaptchaApp class.

        INPUTS: None

        Outputs: None
        """
        # Create a mock environment
        env = CaptchaEnv(source=SyntheticTileSource())
        app = CaptchaApp(env, TILE_SIZE)

        # Check that the attributes are initialized correctly
        assert app.env == env
        app.root.withdraw()
        assert app.tile_size == TILE_SIZE
        assert app.challenge is not None
        assert isinstance(app.selected_tiles, set)
        assert not app.submitted
        app.root.destroy()

    def test_submit(self):
        """
        FUNCTION: test_submit

        DESCRIPTION: This is the unit test for the submit method of the CaptchaApp class.

        INPUTS: None

        Outputs: None
        """
        # Create a mock environment
        env = CaptchaEnv(source=SyntheticTileSource())
        app = CaptchaApp(env, TILE_SIZE)

        # Simulate selecting some tiles and submitting
        app.selected_tiles = {0, 1, 2}
        app.submit()
        app.root.withdraw()

        # Check that the submitted flag is set to True
        assert app.submitted
        app.root.destroy()


    def test_load_challenge(self):
        """
        FUNCTION: load_challenge_test

        DESCRIPTION: This is the unit test for the load_challenge method of the CaptchaApp class.

        INPUTS: None

        Outputs: None
        """
        # Create a mock environment
        env = CaptchaEnv(source=SyntheticTileSource())
        app = CaptchaApp(env, TILE_SIZE)

        # Load a new challenge
        app.load_challenge()
        app.root.withdraw()

        # Check that a new challenge is loaded
        assert app.challenge is not None
        app.root.destroy()


    def test_toggle_tile_selection(self):
        """
        FUNCTION: toggle_tile_selection_test

        DESCRIPTION: This is the unit test for the toggle_tile_selection method of the CaptchaApp class.

        INPUTS: None

        Outputs: None
        """
        # Create a mock environment
        env = CaptchaEnv(source=SyntheticTileSource())
        app = CaptchaApp(env, TILE_SIZE)

        # Simulate toggling tile selection
        app.toggle_tile_selection(0)
        app.root.withdraw()

        assert 0 in app.selected_tiles

        app.toggle_tile_selection(0)
        assert 0 not in app.selected_tiles
        app.root.destroy()