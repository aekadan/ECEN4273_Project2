#############################################################
#   FILE NAME: plot_history.py
#
#   DESCRIPTION: Plots a training run's history JSON (written by
#                Library/train.py) as loss and validation accuracy
#                curves, marking the best epoch (lowest val loss).
#                Run from the repository root:
#                    python Training/plot_history.py
#                    python Training/plot_history.py --history <path> --out <png>
#                Watch a run live (train.py rewrites the history JSON
#                after every epoch) in a second terminal:
#                    python Training/plot_history.py --history <path> --watch
#
#   Authors: Group 2
#############################################################

#############################################################
#   Libraries
#############################################################
import argparse
import json
import sys
from pathlib import Path

import matplotlib
from matplotlib import ticker

#############################################################
#   Global Constants
#############################################################
DEFAULT_HISTORY = Path("checkpoints") / "model_history.json"
DEFAULT_OUT = Path(__file__).resolve().parent / "model_history.png"
DEFAULT_INTERVAL = 2.0      # seconds between reloads in --watch mode
MAX_LABELED_EPOCHS = 20     # label every epoch up to this many


#############################################################
#   FUNCTIONS
#############################################################
def load_history(path):
    """
    FUNCTION NAME: load_history

    DESCRIPTION: Reads a training history JSON file (a list of per-epoch
                 dicts with epoch, train_loss, val_loss, val_accuracy).

    INPUTS: path - path to the history JSON file

    OUTPUTS: list of per-epoch dicts, sorted by epoch
    """
    with open(path, "r", encoding="utf-8") as file:
        history = json.load(file)
    if not history:
        raise ValueError(f"{path} contains no epochs")
    return sorted(history, key=lambda row: row["epoch"])


def set_epoch_ticks(axis, epochs):
    """
    FUNCTION NAME: set_epoch_ticks

    DESCRIPTION: Labels every epoch on short runs (20 or fewer) and lets
                 matplotlib pick whole-number ticks on longer ones, so a
                 100-epoch run stays readable.

    INPUTS: axis - matplotlib Axes
            epochs - list of epoch numbers

    OUTPUTS: None
    """
    if len(epochs) <= MAX_LABELED_EPOCHS:
        axis.set_xticks(epochs)
    else:
        axis.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))


def draw_history(fig, ax_loss, ax_acc, history, title=None):
    """
    FUNCTION NAME: draw_history

    DESCRIPTION: Clears both axes and draws train/val loss and val
                 accuracy per epoch, marking the best (lowest val loss)
                 epoch.

    INPUTS: fig - matplotlib Figure
            ax_loss - axes for the loss curves
            ax_acc - axes for the accuracy curve
            history - list of per-epoch dicts from load_history
            title - optional figure title

    OUTPUTS: None
    """
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    val_acc = [row["val_accuracy"] * 100 for row in history]

    best = min(history, key=lambda row: row["val_loss"])
    best_epoch = best["epoch"]

    ax_loss.clear()
    ax_loss.plot(epochs, train_loss, marker="o", label="Train loss")
    ax_loss.plot(epochs, val_loss, marker="o", label="Val loss")
    ax_loss.axvline(best_epoch, color="gray", linestyle="--", linewidth=1,
                    label=f"Best epoch ({best_epoch})")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("BCE loss")
    ax_loss.set_title("Loss")
    set_epoch_ticks(ax_loss, epochs)
    ax_loss.grid(alpha=0.3)
    ax_loss.legend()

    ax_acc.clear()
    ax_acc.plot(epochs, val_acc, marker="o", color="tab:green",
                label="Val accuracy")
    ax_acc.axvline(best_epoch, color="gray", linestyle="--", linewidth=1,
                   label=f"Best epoch ({best_epoch})")
    ax_acc.annotate(f"{best['val_accuracy'] * 100:.2f}%",
                    (best_epoch, best["val_accuracy"] * 100),
                    textcoords="offset points", xytext=(6, -6),
                    ha="left", va="top")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Validation accuracy (per tile-class, threshold 0.5)")
    set_epoch_ticks(ax_acc, epochs)
    ax_acc.grid(alpha=0.3)
    ax_acc.legend(loc="lower right")

    if title:
        fig.suptitle(title)
    fig.tight_layout()


def make_title(history, title=None):
    """
    FUNCTION NAME: make_title

    DESCRIPTION: Returns the given title, or a default one naming how
                 many epochs the history holds.

    INPUTS: history - list of per-epoch dicts
            title - user-supplied title, or None

    OUTPUTS: str
    """
    return title or f"ResNet-18 training run ({len(history)} epochs)"


def plot_history(history, out_path, title=None):
    """
    FUNCTION NAME: plot_history

    DESCRIPTION: Draws the history on a new figure and saves it as a PNG.

    INPUTS: history - list of per-epoch dicts from load_history
            out_path - where to save the PNG
            title - optional figure title

    OUTPUTS: the saved output path
    """
    import matplotlib.pyplot as plt

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))
    draw_history(fig, ax_loss, ax_acc, history, title)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def watch_history(history_path, out_path, interval=DEFAULT_INTERVAL,
                  title=None):
    """
    FUNCTION NAME: watch_history

    DESCRIPTION: Opens a live window that reloads the history JSON every
                 `interval` seconds and redraws when it changes. Waits if
                 the file doesn't exist yet. When the window is closed,
                 saves the last plot as a PNG.

    INPUTS: history_path - history JSON being written by train.py
            out_path - where to save the final PNG
            interval - seconds between reloads
            title - optional figure title

    OUTPUTS: the saved output path, or None if no epochs were seen
    """
    import matplotlib.pyplot as plt

    history_path = Path(history_path)
    plt.ion()
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"Waiting for {history_path} ...")
    plt.show(block=False)
    print(f"Watching {history_path} every {interval:g}s. "
          f"Close the window to stop.")

    last_mtime = None
    history = None
    while plt.fignum_exists(fig.number):
        try:
            mtime = history_path.stat().st_mtime
            if mtime != last_mtime:
                history = load_history(history_path)
                last_mtime = mtime
                draw_history(fig, ax_loss, ax_acc, history,
                             make_title(history, title))
                latest = history[-1]
                print(f"epoch {latest['epoch']}: "
                      f"val_loss {latest['val_loss']:.4f}  "
                      f"val_acc {latest['val_accuracy']:.4f}")
        except (FileNotFoundError, PermissionError, ValueError):
            # Not written yet, locked mid-swap, or empty: try next tick
            pass
        plt.pause(interval)

    if history is None:
        return None
    return plot_history(history, out_path, make_title(history, title))


def parse_args(argv=None):
    """
    FUNCTION NAME: parse_args

    DESCRIPTION: Parses command-line options for the history path, output
                 PNG path, figure title, and live watch mode.

    INPUTS: argv - list of argument strings, or None for sys.argv

    OUTPUTS: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="Plot a training history JSON.")
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY,
                        help=f"history JSON (default: {DEFAULT_HISTORY})")
    parser.add_argument("--out", type=Path, default=None,
                        help="output PNG (default: Training/<history name>.png)")
    parser.add_argument("--title", default=None, help="figure title")
    parser.add_argument("--watch", action="store_true",
                        help="open a live window that updates as training runs")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                        help=f"seconds between reloads with --watch "
                             f"(default: {DEFAULT_INTERVAL:g})")
    args = parser.parse_args(argv)
    if args.interval <= 0:
        parser.error("--interval must be greater than 0")
    if args.out is None:
        args.out = DEFAULT_OUT.with_name(f"{args.history.stem}.png")
    return args


def main(argv=None):
    """
    FUNCTION NAME: main

    DESCRIPTION: Plots the history to a PNG, or watches it live with
                 --watch, and prints where the PNG went.

    INPUTS: argv - list of argument strings, or None for sys.argv

    OUTPUTS: 0 on success, 1 if the history file is missing or empty
    """
    args = parse_args(argv)

    if args.watch:
        out_path = watch_history(args.history, args.out, args.interval,
                                 args.title)
        if out_path is None:
            print("No epochs were written; nothing saved.")
        else:
            print(f"Saved plot to {out_path}")
        return 0

    # Non-interactive backend: no window needed just to save a PNG
    matplotlib.use("Agg")
    try:
        history = load_history(args.history)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")
        return 1
    out_path = plot_history(history, args.out,
                            make_title(history, args.title))
    print(f"Saved plot to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
