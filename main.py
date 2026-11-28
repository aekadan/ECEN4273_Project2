#############################################################
#   FILE NAME: main.py
#
#   DESCRIPTION: This is the command-line entry point for the
#   project. It is a thin layer over Library/:
#
#       python main.py play  [--data Image-Data/test]
#       python main.py bench [--data Image-Data/test]
#                            [--solver oracle random noisy model]
#                            [-n 1000] [--seed 0] [--p 0.9]
#                            [--checkpoint checkpoints/model.pt]
#                            [--device auto]
#
#   Without --data, both commands use synthetic tiles.
#   Training has its own entry point: python -m Library.train
#
#   Authors: Group 2
#############################################################

#############################################################
#   Libraries
#############################################################

import argparse
import sys
from pathlib import Path

#############################################################
#   Files Import
#############################################################

from Library.benchmark import compare_solvers
from Library.tile_sources import FolderTileSource, SyntheticTileSource
from Library.solvers import make_solver
from Library.train import DEFAULT_CHECKPOINT, DEVICE_CHOICES, select_device

#############################################################
#   Global Constants
#############################################################

SOLVER_CHOICES = ("oracle", "random", "noisy", "model")
DEFAULT_SOLVERS = ["oracle", "random", "noisy"]
DEFAULT_N = 1000
DEFAULT_SEED = 0
DEFAULT_P = 0.9

#############################################################
#   FUNCTIONS
#############################################################

def build_source(data):
    """
    FUNCTION NAME: build_source

    DESCRIPTION: Builds the tile source: real images from a
    split folder when data is given, otherwise synthetic tiles.
    Warns when the folder looks like the training split.

    INPUTS: data (str or None, e.g. "Image-Data/test")

    OUTPUTS: tile source (FolderTileSource or SyntheticTileSource)
    """
    if data is None:
        print("Using synthetic tiles (pass --data Image-Data/test for real images).")
        return SyntheticTileSource()

    folder = Path(data)
    if folder.name == "train":
        print("Warning: benchmarking on the training split overstates the model's "
              "results; use Image-Data/test.", file=sys.stderr)
    source = FolderTileSource(folder)
    counts = ", ".join(f"{label}: {len(source.files[label])}" for label in source.classes)
    print(f"Using images from {folder} ({counts}).")
    return source


def format_results(results):
    """
    FUNCTION NAME: format_results

    DESCRIPTION: Formats benchmark results as a text table.

    INPUTS: results (list of BenchmarkResult)

    OUTPUTS: str
    """
    header = f"{'solver':<8} {'n':>6} {'pass rate':>10} {'tile acc':>9} {'precision':>10} {'recall':>7} {'ms/chal':>8}"
    lines = [header, "-" * len(header)]
    for result in results:
        lines.append(f"{result.solver_name:<8} {result.n:>6} {result.pass_rate:>10.1%} "
                     f"{result.tile_accuracy:>9.3f} {result.precision:>10.3f} "
                     f"{result.recall:>7.3f} {result.seconds_per_challenge * 1000:>8.2f}")
    return "\n".join(lines)


def run_bench(args):
    """
    FUNCTION NAME: run_bench

    DESCRIPTION: Runs every requested solver on the same
    seeded challenges and prints the results table.

    INPUTS: args (argparse.Namespace from the bench command)

    OUTPUTS: list of BenchmarkResult
    """
    source = build_source(args.data)

    solvers = []
    for name in dict.fromkeys(args.solver):   # keep order, drop repeats
        if name == "model":
            if not Path(args.checkpoint).is_file():
                raise FileNotFoundError(f"Checkpoint {args.checkpoint} not found. Train one with "
                                        "python -m Library.train --data Image-Data")
            device = select_device(args.device)
            print(f"Loading {args.checkpoint} on {device}.")
        else:
            device = "cpu"
        solvers.append(make_solver(name, seed=args.seed, p=args.p,
                                   checkpoint=args.checkpoint, device=device,
                                   required_classes=source.classes))

    results = compare_solvers(source, solvers, args.n, args.seed)
    print(f"\n{args.n} challenges, seed {args.seed}" + (f", noisy p={args.p}" if "noisy" in args.solver else ""))
    print(format_results(results))
    return results


def run_play(args):
    """
    FUNCTION NAME: run_play

    DESCRIPTION: Opens the GUI to play challenges by hand.
    The GUI is imported here so the other commands work on
    machines without tkinter.

    INPUTS: args (argparse.Namespace from the play command)

    OUTPUTS: None
    """
    from Library.captcha_env import CaptchaEnv
    from Library.gui import CaptchaApp

    env = CaptchaEnv(build_source(args.data), seed=args.seed)
    CaptchaApp(env).run()


def build_parser():
    """
    FUNCTION NAME: build_parser

    DESCRIPTION: Builds the argparse parser with the play
    and bench commands.

    INPUTS: None

    OUTPUTS: argparse.ArgumentParser
    """
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--data", default=None,
                        help="split folder of class subfolders, e.g. Image-Data/test "
                             "(default: synthetic tiles)")
    shared.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"challenge seed (default: {DEFAULT_SEED})")

    parser = argparse.ArgumentParser(prog="python main.py",
                                     description="Offline 3x3 image CAPTCHA environment.")
    commands = parser.add_subparsers(dest="command")

    play = commands.add_parser("play", parents=[shared], help="solve challenges by hand in a window")
    play.set_defaults(handler=run_play)

    bench = commands.add_parser("bench", parents=[shared], help="score solvers on many challenges")
    bench.add_argument("--solver", nargs="+", choices=SOLVER_CHOICES, default=DEFAULT_SOLVERS,
                       help=f"solvers to compare (default: {' '.join(DEFAULT_SOLVERS)})")
    bench.add_argument("-n", type=int, default=DEFAULT_N,
                       help=f"challenges per solver (default: {DEFAULT_N})")
    bench.add_argument("--p", type=float, default=DEFAULT_P,
                       help=f"per-tile accuracy of the noisy solver (default: {DEFAULT_P})")
    bench.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                       help=f"model checkpoint for --solver model (default: {DEFAULT_CHECKPOINT})")
    bench.add_argument("--device", choices=DEVICE_CHOICES, default="auto",
                       help="device for --solver model (default: auto)")
    bench.set_defaults(handler=run_bench)

    return parser


def main(argv=None):
    """
    FUNCTION NAME: main

    DESCRIPTION: This is the main function for the project.
    Parses the command and runs it. With no command it prints
    the help.

    INPUTS: argv (list of str, or None for sys.argv)

    OUTPUTS: exit code (int)
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    try:
        args.handler(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
