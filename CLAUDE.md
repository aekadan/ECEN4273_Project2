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

The offline 3x3 challenge environment, GUI, and baseline solvers were built by hand and merged into `main` (PR #1, from `story/Create-Recaptcha-Environment`). They replace the earlier plan to apply `offline-captcha-env.patch`; treat that patch as a reference only. The dataset loader landed in PR #2 (`feature/ai-dataset-erica`) and the benchmark in PR #3 (`story/Implement-Benchmark-Test`). The checkpoint helpers in `model.py` (PR #6) and the training loop `Library/train.py` (PR #7, `story/Create-training-loop`) are merged, and PR #5 added the COCO dataset script `prepare_dataset.py`. The framework is PyTorch with a pretrained ResNet-18 (`Library/model.py`).

A real dataset now exists (COCO crops, see `prepare_dataset.py` below), and the model has been trained on it on the RTX 4060 (CUDA, bf16, batch 128, about 4 s per epoch). The best val loss is about 0.10 (around epoch 6-8) with 96.0-96.8% val accuracy (per tile-class decision at 0.5). Longer runs overfit by the loss, yet a 100-epoch run reaches 97.0% val accuracy late; see "Training runs" below. `FolderTileSource`, `ModelSolver`, and the `main.py` command line (`play`, `bench`) are in place, and the README's test-split Results table is filled in: `model_30ep` passes 61.1% of test challenges at the untuned 0.5 thresholds (see "Challenge results" below). `--device cpu` remains the fallback for teammates without a GPU.

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
  - `FolderTileSource(folder_path, classes=None, size=TILE_SIZE)`: real images from one split folder (`Image-Data/test`), one subfolder per class. `classes` defaults to the sorted folder names; `files` maps each class to its sorted image paths (`IMAGE_EXT`, case-insensitive). Raises `FileNotFoundError` for a missing folder or class folder and `ValueError` for an empty class folder, fewer than 2 classes, or a bad size. `get_tile` picks with the passed `rng` only (no state of its own), converts to RGB, resizes to `size`x`size`, and stores the path in `tile.info["source"]`.
  - `CaptchaEnv.new_challenge` copies those paths into `Challenge.sources` when every tile has one (folder source); otherwise `sources` stays `[]`.
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
  - `ModelSolver(checkpoint_path, device="cpu", required_classes=None, thresholds=None)`: loads with `load_checkpoint` (no class-order check, since it looks classes up by name). `required_classes` (e.g. `source.classes`) must all be in the checkpoint, else `ValueError`. `thresholds` starts at `DEFAULT_THRESHOLD` (0.5) per class, then the checkpoint's `thresholds`, then the argument override. `tile_probabilities(tiles)` returns an N x C CPU tensor in checkpoint class order; `solve` returns the sorted indices whose target probability is `>=` the threshold, and raises `ValueError` for an unknown target. It reads only `tiles` and `target`.
  - `make_solver(name, seed=None, p=None, checkpoint=None, device="cpu", required_classes=None)`: builds `"oracle"`, `"random"`, `"noisy"`, or `"model"`. Raises `ValueError` for an unknown name, `"noisy"` without `p`, or `"model"` without `checkpoint`.
  - `solvers.py` now imports torch (through `dataset`/`model`). Only baselines may read `challenge.answer`/`labels`.
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
  - **Checkpoint helpers:** `save_checkpoint(path, model, classes, thresholds=None, extra=None)` writes a `.pt` dict: `arch` (`"resnet18"`), ordered `classes`, `thresholds` (`None` until tuned), CPU `model_state`, plus any `extra` keys (plain Python values only, and they may not overwrite those four), raising `ValueError` if the head size doesn't match `len(classes)` or a threshold names an unknown class. `load_checkpoint(path, expected_classes=None, device="cpu")` returns `(model, checkpoint)` in eval mode, rebuilt with `pretrained=False` (no download), loading with `weights_only=True` and raising `ValueError` on a wrong `arch` or if `expected_classes` differs in content or order. The checkpoint helpers live here, not in `train.py`, so `ModelSolver` never imports training code.
- `Library/train.py`: training loop. Run `python -m Library.train --synthetic` or `--data Image-Data` (reads `train/` and `val/`). Ends with `if __name__ == "__main__": sys.exit(main())`; keep that guard, since Windows DataLoader workers re-import the script.
  - `TrainConfig` dataclass holds every setting (CLI flags will map onto it; `synthetic_train_samples`/`synthetic_val_samples` are config-only). `EpochStats` holds one epoch's `epoch`, `train_loss`, `val_loss`, `val_accuracy` (fraction of tile-class decisions right at 0.5), `lr`, and `seconds` (`compare=False`). Constants include `SYNTHETIC_VAL_SEED_OFFSET` (10000) and `DEVICE_CHOICES`.
  - **Implemented and tested:**
    - `get_train_transform()`: random resized crop (scale 0.8-1.0), horizontal flip, color jitter, `ToTensor`, ImageNet normalization.
    - `select_device(requested)`: `"auto"`/`"cuda"`/`"cpu"` to a `torch.device`; `ValueError` for an unknown name or `"cuda"` without a GPU.
    - `set_seed(seed)`: seeds `random` and torch (and all CUDA devices when present).
    - `validate_config(config)`: `ValueError` unless epochs and batch size are ints >= 1, lr > 0, weight decay >= 0, workers >= 0, synthetic sample counts >= 1.
    - `build_datasets(config)`: synthetic mode returns two `SyntheticTileDataset`s right away (train with `get_train_transform()`, val with `get_transform()` and `seed + SYNTHETIC_VAL_SEED_OFFSET`) and never looks at `data_dir`. Folder mode raises `FileNotFoundError` if `data_dir/train` or `data_dir/val` is missing, then builds `TileDataset`s with val using `classes=<train dataset>.classes` so label positions match. Never reads `test/`.
    - `build_loaders(config, train_dataset, val_dataset, device)`: train shuffles with its own `torch.Generator().manual_seed(config.seed)` (also seeds the workers' augmentation); val doesn't shuffle. Both get `config.batch_size`, `config.num_workers`, `pin_memory` on CUDA only, and `persistent_workers` only when `num_workers > 0`.
    - `build_optimizer(model, config, steps_per_epoch)`: AdamW (`lr`, `weight_decay`) plus `CosineAnnealingLR` with `T_max = epochs * steps_per_epoch`, so the scheduler steps once per batch and reaches ~0 at the end. No warmup.
    - `use_bf16(device)`: True only on CUDA with bf16 support; checks the device first so the GPU query never runs on CPU.
    - `train_one_epoch(...)`: autocast forward (bf16 when enabled), loss on float32 logits, scheduler steps every batch, batch-size-weighted mean loss.
    - `evaluate(...)`: `no_grad`, raw logits with `BCEWithLogitsLoss` (not `predict`), accuracy over every (tile, class) decision at `PREDICTION_THRESHOLD`.
    - `save_history(path, history)`: JSON list of `asdict` per epoch, written to `<name>.tmp` then swapped in with `os.replace` (retried briefly on Windows `PermissionError`) so a live reader never sees a partial file; `history_path_for(out_path)` gives `<stem>_history.json` next to the checkpoint.
    - `fit(config, verbose=True)`: saves the checkpoint (extra: `config` as a dict, `best_epoch`) whenever val loss improves, rewrites `<name>_history.json` after **every** epoch (for live plotting, and so a stopped run keeps its history), and reloads the best at the end and resaves with `history`. Epoch 1 always overwrites an existing checkpoint at `out_path`, so give each run its own `--out`. Default output `checkpoints/model.pt` (gitignored). `EpochStats.lr` is the lr at the start of the epoch.
    - `parse_args(argv)`: `--data`/`--synthetic` mutually exclusive, plus `--epochs`, `--batch-size`, `--lr`, `--weight-decay`, `--device`, `--num-workers`, `--seed`, `--out`, `--no-pretrained`.
    - `main(argv)`: returns 0; on `torch.cuda.OutOfMemoryError` prints a `--batch-size 64` hint, and on `FileNotFoundError`/`ValueError` prints the error; both return 1.
  - bf16 autocast only on CUDA with bf16 support; no GradScaler.
- `prepare_dataset.py` (repo root, by Erica): builds `Image-Data/` from COCO 2017. It expects `Image-Data/original/train2017.zip` and `Image-Data/original/instances_train2017.json`, crops each object's bounding box (non-crowd boxes only, at most one crop per source photo across all classes, `random.Random(42)`), and saves JPEGs (quality 95) for `hydrant` (COCO id 11), `bicycle` (2), `bus` (6), and `traffic_light` (10), 1000 per class split 700/150/150 into `train/`, `val/`, `test/`. It also writes `category_statistics.csv`, `source_licenses.csv` (per-photo Flickr license and URL), and `dataset_manifest.csv` (split, category, filename, source image and annotation ids, license) into `Image-Data/`. Class folder names match `DEFAULT_CLASSES`. Every crop is tightly centered on one object of its class, so there are no background tiles and crop sizes vary (the transforms resize to 224). It has no tests and does not follow the `Library/` import conventions; it is a one-off script.
- `Training/plot_history.py`: standalone matplotlib script (not part of `Library`, no tests) that plots a history JSON as train/val loss and val accuracy with the best epoch marked. `python Training/plot_history.py --history checkpoints/<run>_history.json` saves `Training/<run>_history.png`; add `--watch` (and optionally `--interval`) in a second terminal during training for a live window that redraws each epoch and saves the PNG when closed. Needs `matplotlib` (in `requirements.txt`).
- `main.py`: thin `argparse` command line; `main(argv=None)` returns an exit code (0, or 1 after printing a `FileNotFoundError`/`ValueError`), and with no command it prints the help.
  - `play`: opens `CaptchaApp`. It imports the GUI inside `run_play`, so `bench` works on machines without tkinter.
  - `bench`: runs `compare_solvers` and prints a table (pass rate, tile acc, precision, recall, ms/chal). Options: `--solver` (any of `oracle random noisy model`, default the three baselines), `-n` (1000), `--p` (0.9), `--checkpoint` (`train.DEFAULT_CHECKPOINT`, `checkpoints/model.pt`, which doesn't exist on Zane's machine, so pass e.g. `checkpoints/model_30ep.pt`), `--device` (`auto`).
  - Shared options: `--data` (split folder for `FolderTileSource`; omitted means synthetic tiles) and `--seed` (0). `build_source` warns on stderr when `--data` points at a `train` folder.
- `Unit-Tests/unit_test.py`: 28 pytest tests (counting each parametrized case) covering `main`, `CaptchaEnv` (`new_challenge`, `grade`), `SyntheticTileSource`, `CaptchaApp` (init, submit, load_challenge, toggle), the solvers (oracle, random, noisy, `make_solver`), and the benchmark (`TestBenchmark`: exact oracle and always-wrong scores, counts covering every tile, determinism, invalid `n`, the precision guard via an `EmptySolver` helper, and `compare_solvers` order and fairness). It uses one test class per module.
- `Unit-Tests/test_model_solver.py`: 13 tests: `TestFolderTileSource` (class folders, explicit classes, errors, `get_tile`, rng-only determinism, `Challenge.sources`, shared source in `compare_solvers`) and `TestModelSolver` (selection, thresholds, `tile_probabilities`, class checks, `make_solver("model")`, benchmark). Helpers `make_split` (tiny folder of solid-color images) and `make_checkpoint` (ResNet-18, `pretrained=False`, zero head weights and chosen biases, so every tile gets `sigmoid(bias)`).
- `Unit-Tests/test_main.py`: 7 tests in `TestCommandLine` for `main.py` (help, synthetic bench, folder bench with a model, seeded repeatability, errors, train-split warning, `play` with a fake `Library.gui` in `sys.modules`). These two files sit outside `unit_test.py` so they run without tkinter; `unit_test.py`'s `test_main` calls `main([])`.
- `Unit-Tests/test_dataset.py`: 7 tests for `dataset.py` (synthetic length, image shape, label shape and one-hot labels, reproducibility, multi-label CSV, background-image CSV). They are plain functions rather than a test class, and need `torch` installed.
- `Unit-Tests/test_train.py`: 49 tests (counting parametrized cases) in one `TestTrain` class, covering every `train.py` function and the `model.py` checkpoint helpers: `TrainConfig` defaults, `EpochStats` ignoring `seconds`, `get_train_transform` (shape, normalized, random but seed-repeatable), `select_device` and `use_bf16` (GPU availability faked with `monkeypatch`, so they pass with or without a GPU), `set_seed`, `validate_config` (good config plus 10 bad values), `build_datasets` (synthetic, folders with val sharing train's classes, missing split, unknown val class), `build_loaders` (batch shapes, seeded shuffle vs. ordered val, `pin_memory`/workers options), `build_optimizer` (settings, cosine schedule values, weights update), `train_one_epoch`/`evaluate` (on tiny models, including exact values from a zero-logit `ZeroModel` and short-batch weighting), `save_history`, checkpoint round trip and rejection cases, `fit` (tiny synthetic run, reproducibility), `parse_args`, and `main`. Helpers: `make_config(tmp_path, **overrides)` (CPU, `num_workers=0`, `pretrained=False`, tiny synthetic sizes) and `make_image_folders(root, layout)` (writes a tiny folder dataset). `make_loader(inputs, labels, batch_size)` wraps tensors for loop tests. Optimizer and loop tests use `torch.nn.Linear` instead of ResNet-18; `fit` and checkpoint tests use ResNet-18 with `pretrained=False`. The file runs in about 6 s.
- `.vscode/settings.json`: pytest discovery on `Unit-Tests`, using `.venv`.
- `pytest.ini`: puts the repository root on the import path (`pythonpath = .`) so plain `pytest` works as well as `python -m pytest`, and sets `testpaths = Unit-Tests`. It does not change the capture mode.
- `requirements.txt` and `.gitignore`: dependencies (`pillow`, `pytest`, `matplotlib`, `torch`, `torchvision`) and exclusions for caches, `.venv/`, `.pytest_cache/`, `Image-Data/`, `Training/*.png`, `checkpoints/`, and model weights.

### Known issues

- **GUI tests fail intermittently on Windows (test-only; left unfixed on purpose).** The errors are `_tkinter.TclError`: `Can't find a usable init.tcl` or `tk.tcl`, `invalid command name "tcl_findLibrary"`, or `couldn't read file`. Rerun when you see one, and don't debug project code for it; any other failure is real.
  - Cause: pytest's default fd capture closes the stdin/stdout/stderr handles Tcl stored on the first `Tk()`. Windows can then reuse those handle numbers for Tcl's startup files on a later `Tk()`, and Tcl mistakes them for its old channels (`OpenFileChannel` in `tclWinChan.c`).
  - It is not caused by the GUI code, Pillow, or Tk windows left open, and it doesn't happen outside pytest. A bare `Tk()` under pytest reproduces it, and repeated `CaptchaApp`s without pytest do not.
  - `python -m pytest Unit-Tests --capture=sys` avoids it (0 failures in 80 runs, against 6 in 40 with default capture) and still captures `print()`.
- The GUI tests withdraw and destroy their Tk windows, but `destroy()` is skipped if an assert fails, so a shared fixture would still be useful. They also don't pass a `seed` yet.
- `Image-Data/` is gitignored, so each machine has to build it with `prepare_dataset.py` (it needs the COCO 2017 train zip and annotations, about 19 GB). On the Linux machine it doesn't exist.
- The dataset has no background tiles: every image is a tight crop of one of the 4 classes. Real challenge grids include tiles with none of the classes, so val accuracy here likely overstates real-world performance. Adding background crops (via a `labels.csv` with empty labels) would fix this and is when `pos_weight` becomes relevant.
- On the Linux machine (Python 3.14 `.venv`), `tkinter` is not installed, so `Unit-Tests/unit_test.py` fails at collection (`ModuleNotFoundError` from `Library/gui.py`) and stops the whole run. Install the system Tk package (e.g. `python3-tk` for the matching Python) to fix it, or run `python -m pytest Unit-Tests --ignore=Unit-Tests/unit_test.py` for the other 55 tests.
- Synthetic tiles are a class color with the class name drawn on, so a model learns them from color alone. Use them for wiring and tests only, never as reported results.
- Some `__pycache__/*.pyc` files are tracked despite `.gitignore` (`git rm --cached` them).
- The `origin/story/AI-model` remote branch (with a literal `origin/` prefix) was merged as PR #6 and can be deleted.
- PR #6 committed `__pycache__/*.pyc` files, so they show as modified after every test run.

### Training runs

All on `Image-Data` with the defaults (pretrained, lr 3e-4, weight decay 1e-4, batch 128, seed 0) on the RTX 4060. Checkpoints and histories are in the gitignored `checkpoints/`; val numbers, not test.

| Run | Best epoch | Best val loss | Val acc at best | Notes |
|---|---|---|---|---|
| `model_10ep` | 8 | 0.1033 | 96.54% | Val loss flat from epoch 3; train loss keeps falling |
| `model_30ep` | 7 | 0.1014 | 96.83% | Val loss rises to about 0.125 after epoch 8 and stays there: clear overfitting |
| `model_100ep` | 6 | 0.1040 | 96.00% | Val loss is noisy (0.13-0.20) after epoch 10 and settles at about 0.135; train loss reaches 0.0002. Val accuracy climbs after epoch 45 to 96.8-97.0%, peaking at 97.04% at epoch 60 |

Takeaways:
- Val loss is best at epoch 6-8 in every run; past that the model overfits by the loss.
- Val **accuracy** keeps improving late in long runs (97.0% at epoch 60 vs. 96.0% at the best-loss epoch 6): the late model makes more correct decisions but is overconfident on its mistakes, which BCE punishes. Since pass rate depends on decisions, selecting the checkpoint by val loss (what `fit` does) may not maximize pass rate. Consider tracking or selecting by val accuracy (or by pass rate after the threshold sweep), or saving the last epoch as well, before picking a final model.
- Lowering val loss itself needs more data, background tiles, or stronger regularization.

### Challenge results

`python main.py bench --data Image-Data/test -n 1000 --seed 0 --solver oracle random noisy model --checkpoint checkpoints/<run>.pt --device cuda` on the RTX 4060, all thresholds 0.5. The raw output is in `Training/Challenge_Results.txt` (the `model_30ep` run); this is what the README Results table reports.

| Solver | Pass rate | Tile acc | Precision | Recall | ms/chal |
|---|---|---|---|---|---|
| oracle | 100.0% | 1.000 | 1.000 | 1.000 | 3.6 |
| random | 0.2% | 0.497 | 0.383 | 0.488 | 3.6 |
| noisy (p=0.9) | 40.3% | 0.902 | 0.854 | 0.902 | 3.6 |
| `model_10ep` | 60.5% | 0.944 | 0.961 | 0.892 | 13.5 |
| `model_30ep` | 61.1% | 0.946 | 0.958 | 0.899 | 13.3 |
| `model_100ep` | 59.2% | 0.940 | 0.951 | 0.889 | 14.5 |

Takeaways:
- Pass rate follows tile accuracy to the 9th power (0.946^9 ≈ 61%).
- Test tile accuracy (94.6%) is below val accuracy (96.8%) because val accuracy counts all 4 class decisions per tile, while the challenge only checks the target class.
- Recall is below precision for every checkpoint: most failures are missed targets, so the 0.5 threshold is too strict. The threshold sweep is the next lever.
- `model_100ep` scores lowest despite its late val-accuracy gains, which so far favors selecting checkpoints by val loss.
- `ms/chal` includes building and grading the challenge (about 3.6 ms, the baselines' cost), so model inference is about 10 ms per 9 tiles.

## Roadmap

In planned order:

`FolderTileSource`, the `main.py` command line (`play`, `bench`), `ModelSolver`, and the first README test-split Results table are done (see Current state). Remaining:

1. **Thresholds**: sweep per-class thresholds on the val split (`FolderTileSource("Image-Data/val")`, never `test/`) to maximize full-challenge pass rate (not tile accuracy, since pass rate is roughly tile accuracy to the 9th power), and store them with the checkpoint. Recall (0.899) is below precision (0.958) at 0.5, so expect lower thresholds to help. Then rerun the test benchmark and update the README Results table.
2. **Training improvements**: settle the checkpoint-selection metric (the test benchmark so far favors val loss: `model_30ep` 61.1% vs. `model_100ep` 59.2%; recheck after the threshold sweep), background tiles in the dataset plus `pos_weight` in the loss, and a `main.py train` subcommand that forwards to `train.fit`. Don't add early stopping on val loss until the selection question is settled.
3. **Confusion matrices** for the README test results (`bench` doesn't produce them; needs a small script or a `main.py` option).
4. **`render_grid`**: finish it (paste the 9 tiles into one image) or remove it.
5. **Later**: a GUI option to highlight a solver's picks, for inspecting model mistakes.

Benchmark on `Image-Data/test` only, never on `train/`. The COCO data is fine for academic use (annotations CC BY 4.0; photos carry their Flickr licenses, recorded in `source_licenses.csv`).

Update the Current state and Roadmap sections as items land.

## Intended layout

```
ECEN4273_Project2/
├── Image-Data/          # gitignored; built by prepare_dataset.py: train/ val/ test/, one subfolder per class, plus CSV reports
├── Library/             # package: environment, tile sources, GUI, solvers, benchmark, model, training
├── Training/            # plot_history.py (tracked) and its PNGs (gitignored)
├── Unit-Tests/          # pytest tests: unit_test.py (one Test class per module), test_dataset.py, test_train.py
├── checkpoints/         # gitignored; <run>.pt and <run>_history.json from Library.train
├── main.py              # command-line entry point
├── prepare_dataset.py   # one-off COCO dataset builder
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
- If pytest can't access its default temp folder (`PermissionError` on `pytest-of-<user>`), pass `--basetemp=<some writable folder>`. On Zane's Windows machine this happens, and `--basetemp=.pytest_cache/tmp` works (the session scratchpad under `AppData\Local\Temp\claude` is not visible to the Store Python).
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

The repo is `aekadan/ecen4273_project2` on GitHub, owned by Aek. Work on a branch and open a pull request rather than pushing to `main`. PRs #1 (environment), #2 (dataset loader), #3 (benchmark), #5 (COCO dataset script), #6 (checkpoint helpers), and #7 (training loop) are merged into `main`.
