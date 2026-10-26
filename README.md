# ECEN 4273 – Project 2: CAPTCHA Image Recognition

**Authors:** Group 2

This project trains an image classification model to recognize the objects shown in image-based CAPTCHA challenges (for example, "select all images containing a traffic light"). Given a challenge prompt and a grid of tiles, the model decides which tiles match the prompt and picks them to pass the test.

> **Academic use only.** This project is coursework for ECEN 4273. It is meant to show how robust (or not) image CAPTCHAs are against modern computer vision. Do not use it against live websites or services. Getting around their bot protection may break their Terms of Service and the law.

---

## Table of Contents

- [Project Overview](#project-overview)
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

## Repository Structure

```
ECEN4273_Project2/
├── Image-Data/     # Training, validation, and test images
├── Library/        # Project modules (data loading, model, training, inference)
├── Unit-Tests/     # Unit tests for the project modules
├── main.py         # Entry point for the project
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip
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

Place the image data in `Image-Data/` using one subfolder per class:

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

Run the project from the repository root:

```bash
python main.py
```

_Add the command-line options for training, evaluation, and inference here as they are built._

---

## Testing

Unit tests are in `Unit-Tests/`. To run them:

```bash
python -m pytest Unit-Tests
```

---

## Results

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
| Zane Franics |
| Erica Saenz |
| Aek Nithianandan |