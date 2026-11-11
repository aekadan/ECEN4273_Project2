#############################################################
#   FILE NAME: benchmark.py
#
#   DESCRIPTION: This is the benchmark file which runs a solver 
#   over many challenges and reports pass rate and 
#   per-tile accuracy.
#
#   Authors: Group 2
#############################################################

#############################################################
#   Libraries
#############################################################
from dataclasses import dataclass, field
import time

#############################################################
#   Files Import
#############################################################
from .captcha_env import CaptchaEnv

#############################################################
#   Constants
#############################################################

#############################################################
#   CLASSES
#############################################################
@dataclass
class BenchmarkResult:
    """
    CLASS NAME: BenchmarkResult

    DESCRIPTION: This class stores the totals from a benchmark run.
    The counts are summed over all n challenges.

    INPUTS: solver_name, n, passes, true_positives, false_positives,
    false_negatives, true_negatives, seconds_per_challenge

    OUTPUTS: pass_rate, tile_accuracy, precision, recall
    """
    solver_name: str
    n: int
    passes: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    seconds_per_challenge: float = field(compare=False)

    @property
    def pass_rate(self):
        """
        FUNCTION NAME: pass_rate

        DESCRIPTION: This function returns the pass rate of the benchmark.

        INPUTS: None

        OUTPUTS: float
        """
        return self.passes / self.n

    @property
    def tile_accuracy(self):
        """
        FUNCTION NAME: tile_accuracy

        DESCRIPTION: This function returns the tile accuracy of the benchmark.

        INPUTS: None

        OUTPUTS: float
        """
        return (self.true_positives + self.true_negatives) / (
            self.true_positives
            + self.false_positives
            + self.false_negatives
            + self.true_negatives
        )

    @property
    def precision(self):
        """
        FUNCTION NAME: precision

        DESCRIPTION: This function returns the precision of the benchmark.

        INPUTS: None

        OUTPUTS: float
        """
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self):
        """
        FUNCTION NAME: recall

        DESCRIPTION: This function returns the recall of the benchmark.

        INPUTS: None

        OUTPUTS: float
        """
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

#############################################################
#   FUNCTIONS
#############################################################

def run_benchmark(env, solver, n):
    """
    FUNCTION NAME: run_benchmark

    DESCRIPTION: This function runs a solver on n challenges from env
    and sums the grading results.

    INPUTS: env (CaptchaEnv), solver (Solver), n (positive int)

    OUTPUTS: BenchmarkResult
    """
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError("n must be a positive integer.")

    passes = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    start = time.perf_counter()
    for _ in range(n):
        challenge = env.new_challenge()
        selection = solver.solve(challenge)
        grade = env.grade(challenge, selection)

        if grade.passed:
            passes += 1
        true_positives += grade.true_positives
        false_positives += grade.false_positives
        false_negatives += grade.false_negatives
        true_negatives += grade.true_negatives
    seconds_per_challenge = (time.perf_counter() - start) / n

    return BenchmarkResult(
        solver_name=solver.name,
        n=n,
        passes=passes,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_negatives=true_negatives,
        seconds_per_challenge=seconds_per_challenge,
    )


def compare_solvers(source, solvers, n, seed):
    """
    FUNCTION NAME: compare_solvers

    DESCRIPTION: This function runs each solver 
    on the same challenges by giving each one a 
    fresh env with the same seed.

    INPUTS: source, solvers, n, seed

    OUTPUTS: results (list of BenchmarkResult)
    """
    results = []

    for solver in solvers:
        env = CaptchaEnv(source, seed=seed)
        result = run_benchmark(env, solver, n)
        results.append(result)
    
    return results
