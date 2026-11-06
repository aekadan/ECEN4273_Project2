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
| Benchmark runner for measuring pass rates | Planned |
| Tile source for real images | Planned |
| Command-line options in `main.py` | Planned |
| Dataset, model, and training | Not started |

---

## Repository Structure

```
ECEN4273_Project2/
├── Image-Data/             # Training, validation, and test images (empty for now)
├── Library/                # Project package
│   ├── __init__.py
│   ├── captcha_env.py      # Challenge generator and grader
│   ├── gui.py              # Tkinter window for playing challenges by hand
│   ├── solvers.py          # Baseline solvers and make_solver
│   └── tile_sources.py     # Tile image sources (synthetic placeholder tiles for now)
├── Unit-Tests/             # Unit tests for the project modules
├── main.py                 # Entry point for the project (no commands yet)
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip
- tkinter, for the GUI (included with the standard Windows installer; on Linux it may need a separate package such as `python3-tk`)
- (Optional) A CUDA-capable GPU for faster training

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

---

## Dataset

No dataset has been chosen yet, and the code can't read real images yet. For now, challenges use synthetic placeholder tiles (a solid color with the class name drawn on) for four classes: `bus`, `bicycle`, `traffic_light`, and `hydrant`.

Once the real-image tile source is built, the image data will go in `Image-Data/` using one subfolder per class:

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
| Architecture    | _TBD_ (e.g. custom CNN or fine-tuned ResNet / EfficientNet) |
| Framework       | _TBD_ (e.g. PyTorch or TensorFlow) |
| Input size      | _TBD_   |
| Loss function   | _TBD_ (e.g. cross-entropy) |
| Optimizer       | _TBD_ (e.g. Adam) |
| Augmentation    | _TBD_ (e.g. flips, rotations, color jitter) |

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

_Add the command-line options for benchmarking, training, evaluation, and inference here as they are built._

---

## Testing

Unit tests are in `Unit-Tests/`. To run them:

```bash
python -m pytest Unit-Tests
```

**Known issue on Windows:** the GUI tests sometimes fail with a `TclError` such as `Can't find a usable init.tcl`. This comes from a clash between pytest's default output capture and Tcl, and it only affects test runs, not the application. Rerun the tests when it happens, or avoid it with:

```bash
python -m pytest Unit-Tests --capture=sys
```

---

## Results

No model has been trained yet, so these are still to be measured.

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