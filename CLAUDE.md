# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

ECEN 4273 Project 2 (Group 2: Zane Francis, Erica Saenz, Aek Nithianandan). An academic Python project that trains an image classifier to solve reCAPTCHA-style 3x3 image-grid challenges ("select all images with a traffic light") and measures how often it passes.

Planned pipeline (see README.md):

1. **Preprocess**: split the challenge image into 9 tiles, resize and normalize each.
2. **Classify**: run each tile through a CNN that predicts its object class.
3. **Select**: keep the tiles whose confidence for the target class is above a threshold.
4. **Evaluate**: score the selection against ground truth for per-tile accuracy and full-challenge pass rate.

**Academic use only.** All solving and benchmarking happens against local, offline data. Never write code that targets, scrapes, or submits to live CAPTCHA services or real websites.

## Current state

The offline 3x3 challenge environment, GUI, and baseline solvers were built by hand and merged into `main` (PR #1, from `story/Create-Recaptcha-Environment`). They replace the earlier plan to apply `offline-captcha-env.patch`; treat that patch as a reference only. The benchmark is on `story/Implement-Benchmark-Test`. No dataset, model, or training code exists yet, and the framework is undecided (PyTorch with a pretrained ResNet has been suggested). The README's Dataset, Model, and Results tables are still _TBD_.

### Done

- `Library/__init__.py`: makes `Library` a package.
- `Library/captcha_env.py`
  - Constants `GRID_ROWS`, `GRID_COLS`, `NUM_TILES` (module-level, not attributes of `CaptchaEnv`).
  - `Challenge` dataclass: `id`, `target`, `tiles` (PIL images), `labels`, `sources`, plus `answer` (indices matching the target) and `prompt` ("Select all images with busses."). Tiles are indexed 0-8, row by row from the top-left.
  - `GradeResult` dataclass: `passed`, TP/FP/FN/TN counts, and `tiles_correct`.
  - `CaptchaEnv(source, min_targets=2, max_targets=5, seed=None)`: `new_challenge()` returns a `Challenge`; `grade(challenge, selection)` takes a **list** of unique ints 0-8 and returns a `GradeResult`. A challenge passes only with zero false positives and zero false negatives.
  - `render_grid(challenge, tile_size)`: **stub only** (docstring, returns `None`).
- `Library/tile_sources.py`
  - `TILE_SIZE = 100`, `IMAGE_EXT`, `DEFAULT_CLASSES` (`bus`, `bicycle`, `traffic_light`, `hydrant`).
  - `SyntheticTileSource(classes=None, size=TILE_SIZE)`: placeholder tiles (a solid color per class with the class name drawn on). The tile-source interface is `.classes` plus `get_tile(label, rng)` returning a PIL image.
- `Library/gui.py`: `CaptchaApp(env, tile_size=TILE_SIZE)`, a tkinter window for playing challenges by hand.
  - `__init__` builds the window and calls `load_challenge()` as its last step, so `app.challenge` is set once the constructor returns. It does not call `mainloop()`; `run()` does.
  - `load_challenge()` fetches a new challenge, converts its tiles to `ImageTk.PhotoImage` (references kept in `photo_image_refs`), and resets the selection and the `submitted` flag.
  - `toggle_tile_selection(i)` adds or removes `i` from `selected_tiles` and sets a blue border. It is ignored after submitting.
  - `submit()` grades the sorted selection, updates `score_attempts`/`score_correct` and the result label, and returns the `GradeResult` (or `None` if already submitted). It does **not** clear the selection; `load_challenge()` does.
- `Library/solvers.py`: baseline solvers. The contract is `solve(challenge)`, returning a list of unique tile indices (ints 0-8) that `env.grade` accepts, plus a `name` attribute.
  - `Solver(name, seed=None)`: base class. Stores `name` and its own `rng` (never the env's, so running a solver can't change the challenge sequence). Its `solve` raises `NotImplementedError`.
  - `OracleSolver()`: returns `challenge.answer`, so it always passes.
  - `RandomSolver(seed=None)`: selects each tile on a coin flip, so it passes about 1/512 of the time.
  - `NoisySolver(p, seed=None)`: gets each tile right with probability `p`, so it passes about p^9 of the time. `p` is required and must be from 0 to 1. `p=1.0` always returns the answer, and `p=0.0` returns exactly the wrong tiles.
  - `make_solver(name, seed=None, p=None)`: builds a solver from `"oracle"`, `"random"`, or `"noisy"`. Raises `ValueError` for an unknown name or for `"noisy"` without `p`.
  - Only baselines may read `challenge.answer`/`labels`. The trained CNN will be another `Solver` that uses only `tiles` and `target`, and its class names must match `challenge.target`.
- `Library/benchmark.py`: runs a solver over many challenges and scores it.
  - `BenchmarkResult` dataclass: stores raw counts summed over all challenges (`solver_name`, `n`, `passes`, TP/FP/FN/TN) plus `seconds_per_challenge`. The rates are computed `@property`s: `pass_rate`, `tile_accuracy`, `precision`, and `recall`. `precision` (and `recall`) return `0.0` when their denominator is 0, e.g. a solver that never selects a tile. `seconds_per_challenge` uses `field(compare=False)`, so two results with the same counts compare `==` regardless of timing.
  - `run_benchmark(env, solver, n)`: calls `new_challenge`, `solve`, then `grade` `n` times and returns a `BenchmarkResult`. Raises `ValueError` unless `n` is a positive int (`bool` is rejected). It does not catch `grade`'s errors, so a bad solver selection fails loudly.
  - `compare_solvers(source, solvers, n, seed)`: builds a **fresh** `CaptchaEnv(source, seed=seed)` for each solver so every solver sees the same challenges, and returns a list of results in the same order as `solvers`. Solvers keep their own `rng` state between calls, so pass freshly built solvers for reproducible comparisons.
  - Measured with `seed=1`, `n=2000`: oracle passes 100%; random 0.1% pass, 0.50 tile accuracy; noisy `p=0.9` 38.1% pass, 0.90 tile accuracy.
- `main.py`: entry point. `main()` still just returns 0, and nothing launches the GUI, solvers, or benchmark yet.
- `Unit-Tests/unit_test.py`: 28 pytest tests (counting each parametrized case) covering `main`, `CaptchaEnv` (`new_challenge`, `grade`), `SyntheticTileSource`, `CaptchaApp` (init, submit, load_challenge, toggle), the solvers (oracle, random, noisy, `make_solver`), and the benchmark (`TestBenchmark`: exact oracle and always-wrong scores, counts covering every tile, determinism, invalid `n`, the precision guard via an `EmptySolver` helper, and `compare_solvers` order and fairness). All tests live in this one file, with one test class per module.
- `.vscode/settings.json`: pytest discovery on `Unit-Tests`, using `.venv`.
- `pytest.ini`: puts the repository root on the import path (`pythonpath = .`) so plain `pytest` works as well as `python -m pytest`, and sets `testpaths = Unit-Tests`. It does not change the capture mode.
- `requirements.txt` and `.gitignore`: dependencies (`pillow`, `pytest`) and exclusions for caches, `.venv/`, `Image-Data/`, and model weights.

### Known issues

- **GUI tests fail intermittently on Windows (test-only; left unfixed on purpose).** The errors are `_tkinter.TclError`: `Can't find a usable init.tcl` or `tk.tcl`, `invalid command name "tcl_findLibrary"`, or `couldn't read file`. Rerun when you see one, and don't debug project code for it; any other failure is real.
  - Cause: pytest's default fd capture closes the stdin/stdout/stderr handles Tcl stored on the first `Tk()`. Windows can then reuse those handle numbers for Tcl's startup files on a later `Tk()`, and Tcl mistakes them for its old channels (`OpenFileChannel` in `tclWinChan.c`).
  - It is not caused by the GUI code, Pillow, or Tk windows left open, and it doesn't happen outside pytest. A bare `Tk()` under pytest reproduces it, and repeated `CaptchaApp`s without pytest do not.
  - `python -m pytest Unit-Tests --capture=sys` avoids it (0 failures in 80 runs, against 6 in 40 with default capture) and still captures `print()`.
- The GUI tests withdraw and destroy their Tk windows, but `destroy()` is skipped if an assert fails, so a shared fixture would still be useful. They also don't pass a `seed` yet.
- `Image-Data/` exists locally but is empty and untracked.

## Roadmap

In planned order:

1. **`FolderTileSource`** in `tile_sources.py`: same interface as `SyntheticTileSource`. It reads `Image-Data/<split>/<class>/` using `IMAGE_EXT`, takes classes from the folder names, raises a clear error on empty class folders, and fills `Challenge.sources` with file paths. Like `SyntheticTileSource`, it must keep no random state of its own and use only the `rng` passed to `get_tile`, so `compare_solvers` can share one source across envs. The dataset is not chosen yet; check its license fits academic use. Never benchmark on the training split.
2. **`render_grid`**: finish it (paste the 9 tiles into one image) or remove it.
3. **`main.py` command line** (`argparse`): `play` (launch `CaptchaApp`), `bench --solver <name> -n <count>` (a thin wrapper over `run_benchmark`/`compare_solvers` that prints the `BenchmarkResult` rates), and possibly `show`, with global `--seed` and `--data` (folder source vs. synthetic). Keep it thin. Its benchmark output feeds the README Results table.
4. **Later**: a CNN solver once the dataset and framework are chosen, and a GUI option to highlight a solver's picks, for inspecting model mistakes.

Update the Current state and Roadmap sections as items land.

## Intended layout

```
ECEN4273_Project2/
├── Image-Data/     # train/ val/ test/, one subfolder per class (bicycle/, bus/, traffic_light/, ...)
├── Library/        # package: environment, tile sources, GUI, solvers, benchmark, model, training
├── Unit-Tests/     # pytest tests, currently all in unit_test.py (one Test class per module)
├── main.py         # command-line entry point
└── README.md
```

Put reusable code in `Library/` and keep `main.py` a thin command-line layer over it. Large datasets and model weights should not be committed.

## Setup

Python 3.10+ (the team uses 3.13 on Windows).

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

tkinter ships with the standard Windows Python installer. On Linux it may need a separate package (e.g. `python3-tk`).

## Imports

`Library` is a package, so always run from the repository root.

- Inside `Library/`, import sibling modules relatively: `from .captcha_env import CaptchaEnv`.
- Outside it (`main.py`, tests), import through the package: `from Library.gui import CaptchaApp`.
- Don't add `sys.path` hacks. Mixing `captcha_env` and `Library.captcha_env` loads the module twice with separate class objects.
- Run a library module directly with `python -m Library.<module>`, not `python Library/<module>.py`.

## Testing

Run from the repository root:

```bash
python -m pytest Unit-Tests
```

- Tests are plain pytest classes (`class TestMain:` with bare `assert`), not `unittest.TestCase`.
- pytest only collects methods whose names **start with** `test_` (`test_submit`, not `submit_test`).
- GUI tests create a real `Tk()`: withdraw it immediately and destroy it even when an assert fails (fixture or `try`/`finally`).
- A `TclError` about `init.tcl`, `tk.tcl`, `tcl_findLibrary`, or `couldn't read file` is the known flake (see Known issues): rerun, or use `--capture=sys`.
- Prefer the real `CaptchaEnv(SyntheticTileSource(), seed=...)` over hand-made mocks. `load_challenge` needs real PIL tiles and a `prompt`.
- Add tests for every new `Library/` module, and make sure the suite passes before committing.

## Conventions

Match the existing file style:

- Every `.py` file starts with a `#` banner header:

  ```python
  #############################################################
  #   FILE NAME: <name>.py
  #
  #   DESCRIPTION: <what the file does>
  #
  #   Authors: Group 2
  #############################################################
  ```

  followed by banner sections in this order: `Libraries`, `Files Import`, `Global Constants`, `CLASSES`, `FUNCTIONS` (include only the ones a file needs).

- Every function, method, and class has a docstring in this format (classes use `CLASS NAME:` instead of `FUNCTION NAME:`):

  ```python
  """
  FUNCTION NAME: <name>

  DESCRIPTION: <what it does>

  INPUTS: <args, or None>

  OUTPUTS: <return value, or None>
  """
  ```

- `snake_case` for functions and variables, `PascalCase` for classes, `UPPER_CASE` for module constants, and `if __name__ == "__main__": main()` at the bottom of runnable scripts.
- Top-level folders use `Title-Case-With-Hyphens` (`Image-Data`, `Unit-Tests`).
- Keep README.md in sync when adding commands, the dataset, the model, or results.

## Git

The repo is `aekadan/ecen4273_project2` on GitHub, owned by Aek. Work on a branch and open a pull request rather than pushing to `main`. Current work is on `story/Implement-Benchmark-Test`.
