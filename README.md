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
| ResNet-18 model factory and `predict` | Done |
| Training loop (`train.py`) | Planned |
| Model solver and per-class thresholds | Planned |
| Tile source for real images | Planned |
| Command-line options in `main.py` | Planned |
| Real image dataset | Not chosen |

---

## Repository Structure

```
ECEN4273_Project2/
├── Image-Data/             # Training, validation, and test images (empty for now)
├── Library/                # Project package
│   ├── __init__.py
│   ├── benchmark.py        # Runs solvers over many challenges and scores them
│   ├── captcha_env.py      # Challenge generator and grader
│   ├── dataset.py          # PyTorch datasets: image folders, labels.csv, synthetic tiles
│   ├── gui.py              # Tkinter window for playing challenges by hand
│   ├── model.py            # ResNet-18 classifier (create_model, predict)
│   ├── solvers.py          # Baseline solvers and make_solver
│   └── tile_sources.py     # Tile image sources (synthetic placeholder tiles for now)
├── Unit-Tests/             # Unit tests
│   ├── unit_test.py        # Environment, tile sources, GUI, solvers, benchmark
│   └── test_dataset.py     # Dataset loader
├── main.py                 # Entry point for the project (no commands yet)
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

No dataset has been chosen yet. For now, challenges and `SyntheticTileDataset` use synthetic placeholder tiles (a solid color with the class name drawn on) for four classes: `bus`, `bicycle`, `traffic_light`, and `hydrant`. These tiles are for testing the code only: a model can tell them apart by color alone, so results on them say nothing about real images.

`TileDataset` (in `Library/dataset.py`) already reads real images in either of two layouts. Image data goes in `Image-Data/`, which is not committed, with one subfolder per split:

```
Image-Data/
├── train/
│   ├── bicycle/
│   ├── bus/
│   ├── crosswalk/
│   ├── fire_hydrant/
│   ├── traffic_light/
│   └── ...
├── val/
└── test/
```

For tiles that show several classes, or none of them (background tiles), put a `labels.csv` in the split folder instead, with `;` between labels:

```
filename,labels
img_001.jpg,bus;traffic_light
img_002.jpg,
```

Train on `train/`, tune on `val/`, and benchmark only on `test/`.

| Item          | Details |
|---------------|---------|
| Source        | _TBD_   |
| Classes       | _TBD_   |
| Total images  | _TBD_   |
| Image size    | _TBD_   |
| Split         | _TBD_ (e.g. 70% train / 15% val / 15% test) |

---

## Model

| Item            | Details |
|-----------------|---------|
| Architecture    | ResNet-18 pretrained on ImageNet, with a new linear head (one output per class) |
| Framework       | PyTorch + torchvision |
| Input size      | 224x224 RGB, ImageNet mean/std normalization |
| Output          | Sigmoid probability per class (multi-label) |
| Selection       | Tiles whose probability for the target class is at or above a per-class threshold tuned on `val/` (planned) |
| Loss function   | Binary cross-entropy with logits (planned) |
| Optimizer       | AdamW with a cosine schedule (planned) |
| Training setup  | CUDA if available, bf16 mixed precision, batch 128 (64 if out of memory), `--device cpu` fallback (planned) |
| Augmentation    | Random resized crop, horizontal flip, color jitter on training images only (planned) |

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

---

## Usage

`main.py` has no commands yet, so for now the project is used from Python. Run everything from the repository root, since `Library` is imported as a package.

### Play challenges by hand

```python
from Library.captcha_env import CaptchaEnv
from Library.gui import CaptchaApp
from Library.tile_sources import SyntheticTileSource

env = CaptchaEnv(SyntheticTileSource())
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

_Add the command-line options for benchmarking, training, evaluation, and inference here as they are built._

---

## Testing

Unit tests are in `Unit-Tests/`: `unit_test.py` covers the environment, tile sources, GUI, solvers, and benchmark, and `test_dataset.py` covers the dataset loader (it needs PyTorch installed). To run them:

```bash
python -m pytest Unit-Tests
```

**Known issue on Windows:** the GUI tests sometimes fail with a `TclError` such as `Can't find a usable init.tcl`. This comes from a clash between pytest's default output capture and Tcl, and it only affects test runs, not the application. Rerun the tests when it happens, or avoid it with:

```bash
python -m pytest Unit-Tests --capture=sys
```

---

## Results

No model has been trained yet, so these are still to be measured. Results will be reported on the `test/` split of a real dataset, next to the oracle, random, and noisy baselines on the same seed.

| Metric                      | Value |
|-----------------------------|-------|
| Per-tile accuracy           | _TBD_ |
| Precision / Recall          | _TBD_ |
| Full-challenge pass rate    | _TBD_ |
| Average inference time      | _TBD_ |

_Add training curves, confusion matrices, and a discussion of the results here._

---

## Contributors

**Group 2**, ECEN 4273

| Name |
|------|
| Zane Francis |
| Erica Saenz |
| Aek Nithianandan |