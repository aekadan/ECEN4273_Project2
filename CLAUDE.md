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

The offline 3x3 challenge environment, GUI, and baseline solvers were built by hand and merged into `main` (PR #1, from `story/Create-Recaptcha-Environment`). They replace the earlier plan to apply `offline-captcha-env.patch`; treat that patch as a reference only. The dataset loader landed in PR #2 (`feature/ai-dataset-erica`) and the benchmark in PR #3 (`story/Implement-Benchmark-Test`). The framework is PyTorch with a pretrained ResNet-18 (`Library/model.py`). The training loop (`Library/train.py`) is built on branch `story/Create-training-loop` (not yet merged), but there is no model solver and no real dataset yet, so the README's Results table is still _TBD_. Training targets an NVIDIA RTX 4060 (8 GB): CUDA when available, bf16 autocast, batch 128 (64 if it runs out of memory), and a `--device cpu` fallback for teammates without a GPU.

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
- `Library/dataset.py`: PyTorch datasets for training.
  - Constants `IMAGE_SIZE` (224), `IMAGENET_MEAN`, `IMAGENET_STD`; reuse them rather than retyping the numbers.
  - `get_transform()`: resize to 224x224, `ToTensor`, ImageNet mean/std normalization. This is the eval transform; training augmentation is `train.get_train_transform()`.
  - Both datasets take an optional `transform=` argument (defaults to `get_transform()`).
  - `TileDataset(data_dir, classes=None, transform=None)`: if `data_dir/labels.csv` exists, reads `filename,labels` rows with `;`-separated labels (multi-label tiles and empty-label background tiles are allowed); otherwise reads one subfolder per class (`.png`, `.jpg`, `.jpeg`, one label each). `classes` defaults to the sorted folder names or CSV labels, and unknown classes raise `ValueError`. `__getitem__` returns `(3x224x224 float tensor, multi-hot float label vector)`.
  - `SyntheticTileDataset(num_samples=100, seed=42, transform=None)`: wraps `SyntheticTileSource`; sample `i` has class `classes[i % len(classes)]` and uses `random.Random(seed + i)`, so it is deterministic.
  - It imports `from Library.tile_sources import ...` (absolute), unlike the relative-import convention below.
- `Library/model.py`: `create_model(num_classes=8, pretrained=True)` loads torchvision ResNet-18 (ImageNet weights) and replaces `fc` with an `N`-way linear head; `predict(model, images, device="cpu")` moves the model to `device`, switches to eval, and returns sigmoid probabilities (multi-label, one independent probability per class). Always pass `num_classes=len(dataset.classes)`; the default of 8 does not match the 4 synthetic classes.
  - `save_checkpoint(path, model, classes, thresholds=None, extra=None)` writes a `.pt` dict: `arch` (`"resnet18"`), ordered `classes`, `thresholds` (`None` until tuned), CPU `model_state`, plus any `extra` keys (plain Python values only). It raises `ValueError` if the head size doesn't match `len(classes)`.
  - `load_checkpoint(path, expected_classes=None, device="cpu")` returns `(model, checkpoint)` with the model in eval mode, rebuilt with `pretrained=False` (no download). It loads with `weights_only=True` and raises `ValueError` if `expected_classes` differs in content or order. The checkpoint helpers live here, not in `train.py`, so `ModelSolver` never imports training code.
- `Library/train.py`: training loop. Run `python -m Library.train --synthetic` or `--data Image-Data` (reads `train/` and `val/`).
  - `TrainConfig` dataclass holds every setting (CLI flags map onto it; `synthetic_train_samples`/`synthetic_val_samples` are config-only). `EpochStats` holds one epoch's `train_loss`, `val_loss`, `val_accuracy` (fraction of tile-class decisions right at 0.5), `lr`, and `seconds` (`compare=False`).
  - Pieces: `get_train_transform()` (random resized crop, flip, color jitter), `select_device`, `set_seed`, `validate_config`, `build_datasets` (val is built with the train class list; synthetic val uses `seed + 10000`), `build_loaders` (seeded shuffle, pinned memory on CUDA), `build_optimizer` (AdamW + per-batch cosine over all steps, no warmup), `use_bf16`, `train_one_epoch`, `evaluate` (raw logits, not `predict`), `save_history`, `fit(config, verbose=True)`, `parse_args(argv)`, `main(argv)`.
  - `fit` runs every epoch (no early stopping), saves the checkpoint whenever val loss improves, and at the end reloads the best weights and saves again with the full history, so the checkpoint is always the best epoch. It also writes `<name>_history.json` next to the checkpoint. Default output is `checkpoints/model.pt` (gitignored).
  - bf16 autocast only on CUDA with bf16 support; no GradScaler. `main` catches CUDA out-of-memory and suggests `--batch-size 64`, returning 1. The module ends with `sys.exit(main())`.
  - Verified on the RTX 4060: `--synthetic --epochs 3` reaches val accuracy 1.0 in about 1 s per epoch with 4 workers. The first pretrained run downloads ImageNet weights (~45 MB) to the torch cache.
- `main.py`: entry point. `main()` still just returns 0, and nothing launches the GUI, solvers, or benchmark yet.
- `Unit-Tests/unit_test.py`: 28 pytest tests (counting each parametrized case) covering `main`, `CaptchaEnv` (`new_challenge`, `grade`), `SyntheticTileSource`, `CaptchaApp` (init, submit, load_challenge, toggle), the solvers (oracle, random, noisy, `make_solver`), and the benchmark (`TestBenchmark`: exact oracle and always-wrong scores, counts covering every tile, determinism, invalid `n`, the precision guard via an `EmptySolver` helper, and `compare_solvers` order and fairness). It uses one test class per module.
- `Unit-Tests/test_dataset.py`: 7 tests for `dataset.py` (synthetic length, image shape, label shape and one-hot labels, reproducibility, multi-label CSV, background-image CSV). They are plain functions rather than a test class, and need `torch` installed.
- `Unit-Tests/test_train.py`: 14 tests in `TestTrain` (train transform, dataset `transform=`, `select_device`, `build_datasets` for synthetic/folders/missing split, loss decreases, `fit` history and files, reproducibility, bad config, `parse_args`, `main`) and `TestCheckpoint` (round trip, class mismatch). All run on CPU with `pretrained=False`, `num_workers=0`, and tiny synthetic data via the `make_config(tmp_path, ...)` helper; the file takes about 10 s.
- `.vscode/settings.json`: pytest discovery on `Unit-Tests`, using `.venv`.
- `pytest.ini`: puts the repository root on the import path (`pythonpath = .`) so plain `pytest` works as well as `python -m pytest`, and sets `testpaths = Unit-Tests`. It does not change the capture mode.
- `requirements.txt` and `.gitignore`: dependencies (`pillow`, `pytest`, `torch`, `torchvision`) and exclusions for caches, `.venv/`, `Image-Data/`, and model weights.

### Known issues

- **GUI tests fail intermittently on Windows (test-only; left unfixed on purpose).** The errors are `_tkinter.TclError`: `Can't find a usable init.tcl` or `tk.tcl`, `invalid command name "tcl_findLibrary"`, or `couldn't read file`. Rerun when you see one, and don't debug project code for it; any other failure is real.
  - Cause: pytest's default fd capture closes the stdin/stdout/stderr handles Tcl stored on the first `Tk()`. Windows can then reuse those handle numbers for Tcl's startup files on a later `Tk()`, and Tcl mistakes them for its old channels (`OpenFileChannel` in `tclWinChan.c`).
  - It is not caused by the GUI code, Pillow, or Tk windows left open, and it doesn't happen outside pytest. A bare `Tk()` under pytest reproduces it, and repeated `CaptchaApp`s without pytest do not.
  - `python -m pytest Unit-Tests --capture=sys` avoids it (0 failures in 80 runs, against 6 in 40 with default capture) and still captures `print()`.
- The GUI tests withdraw and destroy their Tk windows, but `destroy()` is skipped if an assert fails, so a shared fixture would still be useful. They also don't pass a `seed` yet.
- `Image-Data/` exists locally but is empty and untracked. No real dataset has been chosen.
- Synthetic tiles are a class color with the class name drawn on, so a model learns them from color alone. Use them for wiring and tests only, never as reported results.
- Some `__pycache__/*.pyc` files are tracked despite `.gitignore` (`git rm --cached` them).
- The `origin/story/AI-model` remote branch (with a literal `origin/` prefix) is old and behind `main`.

## Roadmap

In planned order:

1. **`FolderTileSource`** in `tile_sources.py`: same interface as `SyntheticTileSource`. It reads `Image-Data/<split>/<class>/` using `IMAGE_EXT`, takes classes from the folder names, raises a clear error on empty class folders, and fills `Challenge.sources` with file paths. Like `SyntheticTileSource`, it must keep no random state of its own and use only the `rng` passed to `get_tile`, so `compare_solvers` can share one source across envs. The dataset is not chosen yet; check its license fits academic use. Never benchmark on the training split.
2. **`render_grid`**: finish it (paste the 9 tiles into one image) or remove it.
3. **`main.py` command line** (`argparse`): `play` (launch `CaptchaApp`), `bench --solver <name> -n <count>` (a thin wrapper over `run_benchmark`/`compare_solvers` that prints the `BenchmarkResult` rates), and possibly `show`, with global `--seed` and `--data` (folder source vs. synthetic). Keep it thin. Its benchmark output feeds the README Results table.
4. ~~**`train.py`**~~: built (see Current state). Still open: `pos_weight` in the loss for class imbalance once a dataset with background tiles is chosen, and a `main.py train` subcommand that forwards to `train.fit`.
5. **`ModelSolver`** in `solvers.py` (and `make_solver("model")`): load with `model.load_checkpoint(path, expected_classes=...)`, batch the 9 tiles through `get_transform()` and `predict`, and return the sorted indices whose probability for `challenge.target` is at or above that class's threshold (`checkpoint["thresholds"]`, falling back to 0.5 while it is `None`). It may read only `tiles` and `target`. Class names must match `challenge.target` (the synthetic source uses `hydrant`, not `fire_hydrant`).
6. **Thresholds**: sweep per-class thresholds on the val split to maximize full-challenge pass rate (not tile accuracy, since pass rate is roughly tile accuracy to the 9th power), and store them with the checkpoint.
7. **Later**: fill the README Results table (oracle, random, noisy, and model on the same seed and test split), and a GUI option to highlight a solver's picks, for inspecting model mistakes.

Update the Current state and Roadmap sections as items land.

## Intended layout

```
ECEN4273_Project2/
├── Image-Data/     # train/ val/ test/, one subfolder per class (bicycle/, bus/, traffic_light/, ...)
├── Library/        # package: environment, tile sources, GUI, solvers, benchmark, model, training
├── Unit-Tests/     # pytest tests: unit_test.py (one Test class per module) and test_dataset.py
├── main.py         # command-line entry point
└── README.md
```

Put reusable code in `Library/` and keep `main.py` a thin command-line layer over it. Large datasets and model weights should not be committed.

Design references: the architecture is drawn in `architecture/system-architecture.puml` (PlantUML) and `architecture/system-architecture.uxf` (UMLet) in the project files, not in the repo.

## Setup

Python 3.10+ (the team uses 3.13 on Windows).

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

For GPU training, install the CUDA build of PyTorch before the requirements: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`. The plain PyPI wheel on Windows is CPU-only.

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
- Training tests must stay CPU-only with `pretrained=False` and `num_workers=0` so they need no GPU and no download.
- If pytest can't access its default temp folder (`PermissionError` on `pytest-of-<user>`), pass `--basetemp=<some writable folder>`.
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

The repo is `aekadan/ecen4273_project2` on GitHub, owned by Aek. Work on a branch and open a pull request rather than pushing to `main`. PRs #1 (environment), #2 (dataset loader), and #3 (benchmark) are merged into `main`.
