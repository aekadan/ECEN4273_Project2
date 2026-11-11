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
import pytest

#############################################################
#   Files Import
#############################################################
from Library.gui import CaptchaApp
from Library.captcha_env import CaptchaEnv, Challenge, NUM_TILES
from Library.tile_sources import SyntheticTileSource, DEFAULT_CLASSES, TILE_SIZE
from Library.solvers import Solver, OracleSolver, RandomSolver, NoisySolver, make_solver
from Library.benchmark import run_benchmark, BenchmarkResult, compare_solvers
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

class TestSolver:
    """
    CLASS NAME: TestSolver

    DESCRIPTION: This class contains unit tests for the solver functions.
    """
    def test_oracle_solver(self):
        """
        FUNCTION: test_oracle_solver

        DESCRIPTION: This is the unit test for the OracleSolver class.

        INPUTS: None

        Outputs: None
        """
        solver = OracleSolver()
        challenge = Challenge(id="test", target="cat", tiles=["cat", "dog", "cat", "dog", "cat", "dog", "cat", "dog", "cat"], labels=["cat", "dog"])
        selected_indices = solver.solve(challenge)
        
        # Check that the selected indices match the correct answer
        assert selected_indices == challenge.answer

    def test_random_solver(self):
        """
        FUNCTION: test_random_solver

        DESCRIPTION: This is the unit test for the RandomSolver class.

        INPUTS: None

        Outputs: None
        """
        solver = RandomSolver(seed=42)
        challenge = Challenge(id="test", target="cat", tiles=["cat", "dog", "cat", "dog", "cat", "dog", "cat", "dog", "cat"], labels=["cat", "dog"])
        selected_indices = solver.solve(challenge)
        
        # Check that the selected indices are a subset of the tile indices
        assert all(index in range(NUM_TILES) for index in selected_indices)

    def test_noisy_solver(self):
        """
        FUNCTION: test_noisy_solver

        DESCRIPTION: This is the unit test for the NoisySolver class.

        INPUTS: None

        Outputs: None
        """
        solver = NoisySolver(p=0.5, seed=42)
        challenge = Challenge(id="test", target="cat", tiles=["cat", "dog", "cat", "dog", "cat", "dog", "cat", "dog", "cat"], labels=["cat", "dog"])
        selected_indices = solver.solve(challenge)
        
        # Check that the selected indices are a subset of the tile indices
        assert all(index in range(NUM_TILES) for index in selected_indices)

    def test_make_solver(self):
        """
        FUNCTION: test_solver_factory

        DESCRIPTION: This is the unit test for the solver_factory function.

        INPUTS: None

        Outputs: None
        """
        oracle_solver = make_solver("oracle")
        assert isinstance(oracle_solver, OracleSolver)

        random_solver = make_solver("random", seed=42)
        assert isinstance(random_solver, RandomSolver)

        noisy_solver = make_solver("noisy", p=0.5, seed=42)
        assert isinstance(noisy_solver, NoisySolver)

        # Test for unknown solver name
        try:
            unknown_solver = make_solver("unknown")
            assert False, "Expected ValueError for unknown solver name"
        except ValueError:
            pass

class EmptySolver(Solver):
    """
    CLASS NAME: EmptySolver

    DESCRIPTION: This test helper never selects a tile. It is used to
    check the precision guard when the solver selects nothing.

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
        super().__init__("empty")

    def solve(self, challenge):
        """
        FUNCTION NAME: solve

        DESCRIPTION: This returns an empty selection for every challenge.

        INPUTS: challenge (Challenge)

        OUTPUTS: empty list
        """
        return []

class TestBenchmark:
    """
    CLASS NAME: TestBenchmark

    DESCRIPTION: This class contains unit tests for the benchmark functions.

    INPUTS: None

    OUTPUTS: None
    """
    SEED = 42

    def make_env(self):
        """
        FUNCTION NAME: make_env

        DESCRIPTION: This builds a seeded environment with synthetic tiles.

        INPUTS: None

        OUTPUTS: CaptchaEnv
        """
        return CaptchaEnv(SyntheticTileSource(), seed=self.SEED)

    def test_run_benchmark(self):
        """
        FUNCTION NAME: test_run_benchmark

        DESCRIPTION: This checks that run_benchmark returns a result
        with the solver's name and the number of challenges.

        INPUTS: None

        OUTPUTS: None
        """
        n = 10
        result = run_benchmark(self.make_env(), RandomSolver(seed=self.SEED), n)

        assert isinstance(result, BenchmarkResult)
        assert result.n == n
        assert result.solver_name == "random"
        assert result.seconds_per_challenge >= 0

    def test_oracle_is_perfect(self):
        """
        FUNCTION NAME: test_oracle_is_perfect

        DESCRIPTION: This checks that the oracle solver passes every
        challenge and gets every tile right.

        INPUTS: None

        OUTPUTS: None
        """
        n = 50
        result = run_benchmark(self.make_env(), OracleSolver(), n)

        assert result.passes == n
        assert result.false_positives == 0
        assert result.false_negatives == 0
        assert result.pass_rate == 1.0
        assert result.tile_accuracy == 1.0
        assert result.precision == 1.0
        assert result.recall == 1.0

    def test_always_wrong_scores_zero(self):
        """
        FUNCTION NAME: test_always_wrong_scores_zero

        DESCRIPTION: This checks that a noisy solver with p=0.0, which
        gets every tile wrong, scores zero on every metric.

        INPUTS: None

        OUTPUTS: None
        """
        result = run_benchmark(self.make_env(), NoisySolver(p=0.0, seed=self.SEED), 50)

        assert result.passes == 0
        assert result.pass_rate == 0.0
        assert result.tile_accuracy == 0.0
        assert result.precision == 0.0
        assert result.recall == 0.0

    def test_counts_cover_every_tile(self):
        """
        FUNCTION NAME: test_counts_cover_every_tile

        DESCRIPTION: This checks that every tile of every challenge is
        counted once, including tiles from challenges that failed.

        INPUTS: None

        OUTPUTS: None
        """
        n = 100
        result = run_benchmark(self.make_env(), RandomSolver(seed=self.SEED), n)
        total = (result.true_positives + result.false_positives
                 + result.false_negatives + result.true_negatives)

        assert total == NUM_TILES * n

    def test_run_benchmark_is_deterministic(self):
        """
        FUNCTION NAME: test_run_benchmark_is_deterministic

        DESCRIPTION: This checks that the same env seed and solver seed
        give equal results. Timing is excluded from the comparison.

        INPUTS: None

        OUTPUTS: None
        """
        first = run_benchmark(self.make_env(), RandomSolver(seed=self.SEED), 100)
        second = run_benchmark(self.make_env(), RandomSolver(seed=self.SEED), 100)

        assert first == second

    @pytest.mark.parametrize("n", [0, -1, 2.5, True])
    def test_invalid_n_raises(self, n):
        """
        FUNCTION NAME: test_invalid_n_raises

        DESCRIPTION: This checks that run_benchmark rejects an n that
        is not a positive integer.

        INPUTS: n (invalid challenge count)

        OUTPUTS: None
        """
        with pytest.raises(ValueError):
            run_benchmark(self.make_env(), OracleSolver(), n)

    def test_empty_selection_precision(self):
        """
        FUNCTION NAME: test_empty_selection_precision

        DESCRIPTION: This checks that precision is 0.0 instead of
        dividing by zero when the solver never selects a tile.

        INPUTS: None

        OUTPUTS: None
        """
        result = run_benchmark(self.make_env(), EmptySolver(), 50)

        assert result.true_positives + result.false_positives == 0
        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.pass_rate == 0.0

    def test_random_tile_accuracy(self):
        """
        FUNCTION NAME: test_random_tile_accuracy

        DESCRIPTION: This checks that the random solver gets about half
        of the tiles right.

        INPUTS: None

        OUTPUTS: None
        """
        result = run_benchmark(self.make_env(), RandomSolver(seed=self.SEED), 1000)

        assert abs(result.tile_accuracy - 0.5) < 0.05

    def test_compare_solvers(self):
        """
        FUNCTION NAME: test_compare_solvers

        DESCRIPTION: This checks that compare_solvers returns one result
        per solver, in the same order as the solvers.

        INPUTS: None

        OUTPUTS: None
        """
        solvers = [OracleSolver(), RandomSolver(seed=self.SEED),
                   NoisySolver(p=0.5, seed=self.SEED)]
        n = 10
        results = compare_solvers(SyntheticTileSource(), solvers, n, self.SEED)

        assert isinstance(results, list)
        assert [r.solver_name for r in results] == [s.name for s in solvers]
        for result in results:
            assert isinstance(result, BenchmarkResult)
            assert result.n == n

    def test_compare_solvers_same_challenges(self):
        """
        FUNCTION NAME: test_compare_solvers_same_challenges

        DESCRIPTION: This checks that every solver sees the same
        challenges. Two identical solvers only give equal results if
        each one gets a fresh env with the same seed.

        INPUTS: None

        OUTPUTS: None
        """
        solvers = [RandomSolver(seed=1), RandomSolver(seed=1)]
        results = compare_solvers(SyntheticTileSource(), solvers, 100, self.SEED)

        assert results[0] == results[1]

    def test_compare_solvers_matches_run_benchmark(self):
        """
        FUNCTION NAME: test_compare_solvers_matches_run_benchmark

        DESCRIPTION: This checks that compare_solvers gives the same
        result as calling run_benchmark with a fresh env of the same seed.

        INPUTS: None

        OUTPUTS: None
        """
        compared = compare_solvers(SyntheticTileSource(),
                                   [NoisySolver(p=0.9, seed=1)], 100, self.SEED)
        direct = run_benchmark(self.make_env(), NoisySolver(p=0.9, seed=1), 100)

        assert compared == [direct]

    def test_compare_solvers_empty_list(self):
        """
        FUNCTION NAME: test_compare_solvers_empty_list

        DESCRIPTION: This checks that comparing no solvers returns an
        empty list.

        INPUTS: None

        OUTPUTS: None
        """
        assert compare_solvers(SyntheticTileSource(), [], 10, self.SEED) == []
