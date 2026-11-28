# ECEN 4273 – Project 2: CAPTCHA Image Recognition

**Authors:** Group 2

This project trains an image classification model to recognize the objects shown in image-based CAPTCHA challenges (for example, "select all images containing a traffic light"). Given a challenge prompt and a grid of tiles, the model decides which tiles match the prompt and picks them to pass the test.

> **Academic use only.** This project is coursework for ECEN 4273. It is meant to show how robust (or not) image CAPTCHAs are against modern computer vision. Do not use it against live websites or services. Getting around their bot protection may break their Terms of Service and the law.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Status](#project-status)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Dataset](#dataset)
- [Model](#model)
- [Usage](#usage)
- [Testing](#testing)
- [Results](#results)
- [Contributors](#contributors)

---

## Project Overview

Image CAPTCHAs ask a user to find every tile in a grid that contains a target object class. The project pipeline is:

1. **Preprocess.** Load the CAPTCHA image, split it into tiles, and resize and normalize each tile.
2. **Classify.** Run each tile through a trained convolutional neural network (CNN) that predicts the object class.
3. **Select.** Compare each prediction with the challenge's target class and keep the tiles whose confidence is above a threshold.
4. **Evaluate.** Score the selection against the ground truth to measure the pass rate.

### Goals

- Build a labeled dataset of CAPTCHA-style image tiles.
- Train a model that classifies tiles into the common CAPTCHA categories.
- Reach a high per-tile accuracy and a high full-challenge pass rate.
- Write up what the results mean for CAPTCHA security.

---

## Project Status

All challenges are generated and graded locally, with no connection to any real CAPTCHA service.

| Component | Status |
|-----------|--------|
| Offline 3x3 challenge generator and grader | Done |
| Synthetic placeholder tiles | Done |
| Tkinter window for playing challenges by hand | Done |
| Baseline solvers (oracle, random, noisy) | Done |
| Benchmark runner for measuring pass rates | Done |
| Dataset loader (`TileDataset`, `SyntheticTileDataset`) | Done |
| ResNet-18 model factory, `predict`, and checkpoints | Done |
| Real image dataset (COCO 2017 crops, `prepare_dataset.py`) | Done |
| Training loop (`train.py`) | Done (trained on the COCO crops on an RTX 4060) |
| Training curve plots, including a live view (`Training/plot_history.py`) | Done |
| Model solver and per-class thresholds | Planned |
| Tile source for real images (challenges still use synthetic tiles) | Planned |
| Command-line options in `main.py` | Planned |

---

## Repository Structure

```
ECEN4273_Project2/
├── Image-Data/             # Not committed; built by prepare_dataset.py (train/, val/, test/)
├── Library/                # Project package
│   ├── __init__.py
│   ├── benchmark.py        # Runs solvers over many challenges and scores them
│   ├── captcha_env.py      # Challenge generator and grader
│   ├── dataset.py          # PyTorch datasets: image folders, labels.csv, synthetic tiles
│   ├── gui.py              # Tkinter window for playing challenges by hand
│   ├── model.py            # ResNet-18 classifier, predict, save/load checkpoints
│   ├── solvers.py          # Baseline solvers, ModelSolver, and make_solver
│   ├── tile_sources.py     # Tile sources: synthetic placeholders and real image folders
│   └── train.py            # Training loop (python -m Library.train)
├── Training/
│   └── plot_history.py     # Plots training curves, live or to a PNG
├── Unit-Tests/             # Unit tests
│   ├── unit_test.py        # Environment, tile sources, GUI, solvers, benchmark
│   ├── test_dataset.py     # Dataset loader
│   └── test_train.py       # Training loop and checkpoints
├── checkpoints/            # Not committed; trained models and their history JSON
├── main.py                 # Entry point for the project (no commands yet)
├── prepare_dataset.py      # Builds Image-Data/ from COCO 2017
├── pytest.ini              # Test discovery settings
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip
- tkinter, for the GUI (included with the standard Windows installer; on Linux it may need a separate package such as `python3-tk`)
- PyTorch and torchvision (installed by `requirements.txt`)
- (Optional) An NVIDIA GPU with CUDA for faster training. The team's reference card is an RTX 4060 (8 GB).

### Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd ECEN4273_Project2
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\activate

   # macOS / Linux
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   On Windows, `pip` installs the CPU-only build of PyTorch. To train on an NVIDIA GPU, install the CUDA build first, then the requirements:

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   pip install -r requirements.txt
   ```

   Check it with `python -c "import torch; print(torch.cuda.is_available())"`.

---

## Dataset

The model is trained on object crops from the [COCO 2017](https://cocodataset.org/) training set. `prepare_dataset.py` cuts out the bounding box of each chosen object and sorts the crops into class folders:

| Item          | Details |
|---------------|---------|
| Source        | COCO 2017 `train2017` photos and `instances_train2017.json` annotations |
| Classes       | `bicycle`, `bus`, `hydrant` (COCO "fire hydrant"), `traffic_light` |
| Total images  | 4,000 (1,000 per class) |
| Image size    | Varies (a tight crop of each object's box); resized to 224x224 for the model |
| Split         | 70% train / 15% val / 15% test: 700 / 150 / 150 per class |
| Selection     | Non-crowd boxes, at most one crop per COCO photo, fixed seed (42) |
| License       | COCO annotations are CC BY 4.0; each photo keeps its Flickr Creative Commons license, listed in `source_licenses.csv` |

To rebuild it, put `train2017.zip` and `instances_train2017.json` from the COCO website in `Image-Data/original/` and run:

```bash
python prepare_dataset.py
```

This writes the class folders plus three reports: `dataset_manifest.csv` (where every crop came from), `source_licenses.csv`, and `category_statistics.csv`. `Image-Data/` is not committed, so each teammate builds it locally.

```
Image-Data/
├── original/           # COCO downloads (input)
├── train/
│   ├── bicycle/
│   ├── bus/
│   ├── hydrant/
│   └── traffic_light/
├── val/
├── test/
├── category_statistics.csv
├── dataset_manifest.csv
└── source_licenses.csv
```

Every image is a close crop of one object of its class, so the dataset has no background tiles (tiles with none of the classes). Real challenge grids do have them, so the current scores are likely optimistic.

Challenges in the GUI and benchmark still use synthetic placeholder tiles (a solid color with the class name drawn on) until a tile source for the real images is built. `SyntheticTileDataset` uses the same tiles. They are for testing the code only: a model can tell them apart by color alone, so results on them say nothing about real images.

`TileDataset` (in `Library/dataset.py`) reads either one subfolder per class, as above, or a `labels.csv` in the split folder for tiles that show several classes or none of them, with `;` between labels:

```
filename,labels
img_001.jpg,bus;traffic_light
img_002.jpg,
```

Train on `train/`, tune on `val/`, and benchmark only on `test/`.

---

## Model

| Item            | Details |
|-----------------|---------|
| Architecture    | ResNet-18 pretrained on ImageNet, with a new linear head (one output per class) |
| Framework       | PyTorch + torchvision |
| Input size      | 224x224 RGB, ImageNet mean/std normalization |
| Output          | Sigmoid probability per class (multi-label) |
| Selection       | Tiles whose probability for the target class is at or above a per-class threshold tuned on `val/` (planned) |
| Loss function   | Binary cross-entropy with logits |
| Optimizer       | AdamW (lr 3e-4, weight decay 1e-4) with a per-batch cosine schedule |
| Training setup  | CUDA if available, bf16 mixed precision, batch 128 (64 if out of memory), `--device cpu` fallback |
| Augmentation    | Random resized crop, horizontal flip, color jitter on training images only |
| Checkpoint      | Best epoch by val loss: weights, ordered class list, thresholds (once tuned), config, and history |

```python
from Library.dataset import SyntheticTileDataset
from Library.model import create_model, predict
from torch.utils.data import DataLoader

dataset = SyntheticTileDataset(num_samples=8)
model = create_model(num_classes=len(dataset.classes))
images, labels = next(iter(DataLoader(dataset, batch_size=8)))
probs = predict(model, images)  # shape (8, 4), untrained
```

Always build the model with `num_classes=len(dataset.classes)`, so the head matches the dataset's classes and their order.

### Training

Run from the repository root:

```bash
# Check the wiring on synthetic tiles (results mean nothing about real images)
python -m Library.train --synthetic --epochs 3

# Train on Image-Data/train, validating on Image-Data/val
python -m Library.train --data Image-Data --epochs 10 --out checkpoints/model_10ep.pt

# No GPU, or out of memory
python -m Library.train --data Image-Data --device cpu --batch-size 64
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--data` / `--synthetic` | `Image-Data` | Folder holding `train/` and `val/`, or synthetic tiles instead |
| `--epochs` | 10 | Number of passes over the training data |
| `--batch-size` | 128 | Use 64 if CUDA runs out of memory |
| `--lr`, `--weight-decay` | 3e-4, 1e-4 | AdamW settings |
| `--device` | `auto` | `auto`, `cuda`, or `cpu` |
| `--num-workers` | 4 | DataLoader worker processes (0 loads in the main process) |
| `--seed` | 0 | Seed for weights init, shuffling, and augmentation |
| `--out` | `checkpoints/model.pt` | Checkpoint path (the folder is gitignored) |
| `--no-pretrained` | off | Start from random weights instead of ImageNet |

Each epoch prints the train loss, val loss, val accuracy, and learning rate, with `*` when val loss improves. The checkpoint always holds the best epoch, and `<name>_history.json` next to it (for example `model_10ep_history.json`) holds the per-epoch numbers. The history is rewritten after every epoch, so a run you stop early keeps it. The first run downloads the ImageNet weights (about 45 MB). On the RTX 4060 an epoch takes about 4 seconds.

**Give each run its own `--out`.** The first epoch always saves over whatever checkpoint is at that path, so reusing a name replaces the earlier model.

### Plotting training curves

`Training/plot_history.py` plots train and val loss and val accuracy per epoch, with the best epoch marked. To watch a run live, start it in a second terminal while training:

```bash
python Training/plot_history.py --history checkpoints/model_10ep_history.json --watch
```

The window redraws after every epoch (it checks every 2 seconds; change with `--interval`), and saves `Training/model_10ep_history.png` when you close it. Without `--watch` it just saves the PNG. `--out` and `--title` change the file name and figure title.

To load a trained model:

```python
from Library.model import load_checkpoint, predict

model, checkpoint = load_checkpoint("checkpoints/model_10ep.pt")
print(checkpoint["classes"], checkpoint["best_epoch"])
```

---

## Usage

Run everything from the repository root, since `Library` is imported as a package.

### Command line

```bash
# Play challenges by hand (synthetic tiles, or real test images)
python main.py play
python main.py play --data Image-Data/test

# Compare the baselines on real test images
python main.py bench --data Image-Data/test -n 1000

# Add the trained model (checkpoints/model.pt by default)
python main.py bench --data Image-Data/test -n 1000 --solver oracle random noisy model
```

```
solver        n  pass rate  tile acc  precision  recall  ms/chal
----------------------------------------------------------------
oracle     1000     100.0%     1.000      1.000   1.000     ...
```

| Option | Commands | Default | Meaning |
|--------|----------|---------|---------|
| `--data` | both | synthetic tiles | Split folder with one subfolder per class. Benchmark on `Image-Data/test` only |
| `--seed` | both | 0 | Challenge seed; every solver sees the same challenges |
| `--solver` | bench | `oracle random noisy` | Any of `oracle`, `random`, `noisy`, `model` |
| `-n` | bench | 1000 | Challenges per solver |
| `--p` | bench | 0.9 | Per-tile accuracy of the noisy solver |
| `--checkpoint` | bench | `checkpoints/model.pt` | Model for `--solver model` |
| `--device` | bench | `auto` | `auto`, `cuda`, or `cpu` for the model |

### Play challenges by hand from Python

```python
from Library.captcha_env import CaptchaEnv
from Library.gui import CaptchaApp
from Library.tile_sources import FolderTileSource

env = CaptchaEnv(FolderTileSource("Image-Data/test"))
CaptchaApp(env).run()
```

Click tiles to select them, press **Submit** to grade your selection, and press **Next** for a new challenge. The window keeps a running score.

### Run a solver

```python
from Library.captcha_env import CaptchaEnv
from Library.solvers import make_solver
from Library.tile_sources import SyntheticTileSource

env = CaptchaEnv(SyntheticTileSource(), seed=1)
solver = make_solver("noisy", seed=1, p=0.9)

challenge = env.new_challenge()
result = env.grade(challenge, solver.solve(challenge))
print(result.passed, result.tiles_correct)
```

A challenge passes only if all 9 tiles are right. These baselines set the reference points for the trained model:

| Solver | Behavior | Expected pass rate |
|--------|----------|--------------------|
| `oracle` | Returns the correct answer | 100% |
| `random` | Selects each tile on a coin flip | About 0.2% (1/512) |
| `noisy` | Gets each tile right with probability `p` | About `p`^9 (39% at `p=0.9`) |
| `model` | Selects tiles whose model probability for the target is at or above the class threshold (0.5 until tuned) | Measured below |

The model solver only looks at the tiles and the target, never the answer:

```python
from Library.solvers import ModelSolver
from Library.tile_sources import FolderTileSource

source = FolderTileSource("Image-Data/test")
solver = ModelSolver("checkpoints/model.pt", device="cuda", required_classes=source.classes)
```

### Benchmark solvers

`compare_solvers` runs each solver on the same set of challenges (each one gets a fresh environment built from the same seed) and returns one `BenchmarkResult` per solver:

```python
from Library.benchmark import compare_solvers
from Library.solvers import NoisySolver, OracleSolver, RandomSolver
from Library.tile_sources import SyntheticTileSource

solvers = [OracleSolver(), RandomSolver(seed=1), NoisySolver(p=0.9, seed=1)]
for r in compare_solvers(SyntheticTileSource(), solvers, n=2000, seed=1):
    print(f"{r.solver_name:8} pass={r.pass_rate:.3f} acc={r.tile_accuracy:.3f} "
          f"precision={r.precision:.3f} recall={r.recall:.3f}")
```

```
oracle   pass=1.000 acc=1.000 precision=1.000 recall=1.000
random   pass=0.001 acc=0.501 precision=0.392 recall=0.503
noisy    pass=0.381 acc=0.898 precision=0.849 recall=0.898
```

To benchmark a single solver, use `run_benchmark(env, solver, n)` with a seeded `CaptchaEnv`. Build fresh solvers for each comparison, since a solver's random state carries over between runs.

| Metric | Meaning |
|--------|---------|
| `pass_rate` | Fraction of challenges with every tile right |
| `tile_accuracy` | Fraction of individual tiles right |
| `precision` | Of the tiles selected, the fraction that were targets (0 if nothing was selected) |
| `recall` | Of the target tiles, the fraction that were selected |
| `seconds_per_challenge` | Average time to solve and grade one challenge |

---

## Testing

Unit tests are in `Unit-Tests/`: `unit_test.py` covers the environment, tile sources, GUI, solvers, and benchmark, `test_dataset.py` covers the dataset loader, `test_train.py` covers the training loop and checkpoints, `test_model_solver.py` covers `FolderTileSource` and `ModelSolver`, and `test_main.py` covers the command line (these four need PyTorch installed but not tkinter; the training tests run on the CPU with tiny synthetic data, so they need no GPU or download). To run them:

```bash
python -m pytest Unit-Tests
```

**Known issue on Windows:** the GUI tests sometimes fail with a `TclError` such as `Can't find a usable init.tcl`. This comes from a clash between pytest's default output capture and Tcl, and it only affects test runs, not the application. Rerun the tests when it happens, or avoid it with:

```bash
python -m pytest Unit-Tests --capture=sys
```

If pytest fails with a `PermissionError` on its temp folder (`pytest-of-<user>`), point it at a writable one with `--basetemp=.pytest_cache/tmp`.

---

## Results

### Training (validation split)

All runs use the default settings (ImageNet-pretrained ResNet-18, AdamW lr 3e-4, batch 128, seed 0) on an RTX 4060. Val accuracy counts each (tile, class) yes/no decision at a 0.5 threshold, so it is not a challenge pass rate.

| Run | Epochs | Best epoch | Best val loss | Val accuracy at best epoch |
|-----|--------|------------|---------------|----------------------------|
| `model_10ep` | 10 | 8 | 0.103 | 96.5% |
| `model_30ep` | 30 | 7 | 0.101 | 96.8% |
| `model_100ep` | 100 | 6 | 0.104 | 96.0% |

To see these curves, run `python Training/plot_history.py --history checkpoints/model_100ep_history.json` (or another run's history), which writes `Training/model_100ep_history.png`. Plot images are not committed.

In every run, val loss is lowest around epoch 6-8, while train loss keeps falling toward zero (to 0.0002 by epoch 100). After that, val loss rises to about 0.125 (30 epochs) or 0.135-0.20 (100 epochs), which shows overfitting. The checkpoint keeps the best epoch by val loss, so the saved model is never the overfit one.

Val accuracy tells a different story, though. In the 100-epoch run it is noisy (94-96%) until about epoch 45, then climbs and settles at 96.8-97.0%, peaking at 97.0% at epoch 60, while val loss stays at about 0.135. The late model makes more correct yes/no decisions, but is overconfident on the tiles it still gets wrong, and binary cross-entropy punishes that heavily. A challenge passes on decisions, not loss, so choosing the checkpoint by val loss may not give the best pass rate. The threshold sweep on `val/` will settle which epoch to keep. The difference is about 1 point of tile accuracy, which over 9 tiles is several points of pass rate.

### Challenges (test split)

Still to be measured: this needs the model solver and a tile source for the real images. Results will be reported on the `test/` split, next to the oracle, random, and noisy baselines on the same seed. A rough guide: if every tile were 96.5% right on its own, a 9-tile challenge would pass about 0.965^9 ≈ 73% of the time, before tuning thresholds.

| Metric                      | Value |
|-----------------------------|-------|
| Per-tile accuracy           | _TBD_ |
| Precision / Recall          | _TBD_ |
| Full-challenge pass rate    | _TBD_ |
| Average inference time      | _TBD_ |

_Add confusion matrices and a discussion of the test results here._

---

## Contributors

**Group 2**, ECEN 4273

| Name |
|------|
| Zane Francis |
| Erica Saenz |
| Aek Nithianandan |
