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

An offline 3x3 challenge environment is being built by hand on the `story/Create-Recaptcha-Environment` branch. It replaces the earlier plan to apply `offline-captcha-env.patch`; treat that patch as a reference only. No dataset, model, or training code exists yet, and the framework is undecided (PyTorch with a pretrained ResNet has been suggested). The README's Dataset, Model, and Results tables are still _TBD_.

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
- `main.py`: entry point. `main()` still just returns 0, and nothing launches the GUI yet.
- `Unit-Tests/unit_test.py`: 9 pytest tests covering `main`, `CaptchaEnv` (`new_challenge`, `grade`), `SyntheticTileSource`, and `CaptchaApp` (init, submit, load_challenge, toggle).
- `.vscode/settings.json`: pytest discovery on `Unit-Tests`, using `.venv`.

### Known issues

- Only `test_init` withdraws and destroys its Tk window. The other GUI tests leave Tk instances open, and `test_load_challenge` has failed intermittently with a `_tkinter` error. The planned fix is a shared pytest fixture that builds the app, calls `root.withdraw()`, yields, then calls `root.destroy()`. The GUI tests also don't pass a `seed` yet.
- `__pycache__/` folders are committed by mistake, and there is no `requirements.txt`, `.gitignore`, or `pytest.ini` yet.
- `Image-Data/` exists locally but is empty and untracked.

## Roadmap

In planned order:

1. **Solvers** (`Library/solvers.py`): one contract, where a solver takes a `Challenge` and returns a list of tile indices. Baselines: `random`, `oracle` (returns `challenge.answer`, so it should pass 100%), and `noisy` (correct on each tile with probability p, so pass rate is roughly p^9). The trained CNN will plug in as another solver.
2. **Benchmark** (`Library/benchmark.py`): `run_benchmark(env, solver, n)` reports pass rate and per-tile accuracy (optionally precision/recall from the `GradeResult` counts). Use a seeded env so solvers are compared on the same challenges. Its output feeds the README Results table.
3. **`FolderTileSource`** in `tile_sources.py`: same interface as `SyntheticTileSource`. It reads `Image-Data/<split>/<class>/` using `IMAGE_EXT`, takes classes from the folder names, raises a clear error on empty class folders, and fills `Challenge.sources` with file paths. The dataset is not chosen yet; check its license fits academic use. Never benchmark on the training split.
4. **`render_grid`**: finish it (paste the 9 tiles into one image) or remove it.
5. **`main.py` command line** (`argparse`): `play` (launch `CaptchaApp`), `bench --solver <name> -n <count>`, and possibly `show`, with global `--seed` and `--data` (folder source vs. synthetic). Keep it thin.
6. **Housekeeping**: `requirements.txt` (`pillow`, `pytest`), `.gitignore` (`__pycache__/`, `.venv/`, `.pytest_cache/`, `Image-Data/`, model weights), `git rm -r --cached` the committed `__pycache__` files, an optional `pytest.ini` (`pythonpath = .`, `testpaths = Unit-Tests`), then a PR to `main`.
7. **Later**: a GUI option to highlight a solver's picks, for inspecting model mistakes.

Update the Current state and Roadmap sections as items land.

## Intended layout

```
ECEN4273_Project2/
├── Image-Data/     # train/ val/ test/, one subfolder per class (bicycle/, bus/, traffic_light/, ...)
├── Library/        # package: environment, tile sources, GUI, solvers, benchmark, model, training
├── Unit-Tests/     # pytest tests, files named *_test.py
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
pip install pillow pytest     # until requirements.txt exists
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

The repo is `aekadan/ecen4273_project2` on GitHub, owned by Aek. Work on a branch and open a pull request rather than pushing to `main`. Current work is on `story/Create-Recaptcha-Environment`.
