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

The repo is an early scaffold. No dataset, model, or training code exists yet, and the framework is undecided (PyTorch with a pretrained ResNet has been suggested). The README's Dataset, Model, and Results tables are still _TBD_.

- `main.py`: entry point. `main()` currently just returns 0.
- `Unit-Tests/unit_test.py`: one pytest test asserting `main() == 0`.
- `README.md`: full project description and intended layout.
- `LICENSE.md`: Academic Software License (ASL).
- `.vscode/settings.json`: VS Code test settings (see Testing).

`Image-Data/` and `Library/` are described in the README but do not exist yet, since git does not track empty folders. There is no `requirements.txt` or `.gitignore` yet, and `__pycache__/` folders are committed by mistake.

### Offline challenge simulator (not merged yet)

An offline 3x3 challenge environment has been built but is not in the repo yet. It lives as a git patch (`offline-captcha-env.patch`, applied with `git am`) in the project's shared files. It adds:

- `Library/captcha_env.py`: challenge generator and grader (`CaptchaEnvironment`, `render_grid`).
- `Library/tile_sources.py`: `SyntheticTileSource` (placeholder tiles) and `FolderTileSource` (reads `Image-Data/<split>/<class>/`).
- `Library/solvers.py`: baseline solvers (`random`, `oracle`, `noisy`, manual) via `make_solver`.
- `Library/benchmark.py`: `run_benchmark` for pass-rate measurement.
- `Library/web_app.py`: standard-library web server to take challenges in a browser.
- `main.py` subcommands: `bench`, `serve`, `show`, `play` (global options `--data`, `--seed`, `--max-mistakes`).
- `requirements.txt` (`pillow`, `pytest`), `pytest.ini`, `.gitignore`, and 17 pytest tests.

Once applied, the trained model should plug in as another solver so it can be benchmarked the same way. Update this section when the patch lands.

## Intended layout

```
ECEN4273_Project2/
├── Image-Data/     # train/ val/ test/, one subfolder per class (bicycle/, bus/, traffic_light/, ...)
├── Library/        # project modules: data loading, model, training, inference
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
pip install -r requirements.txt   # once requirements.txt exists
```

## Testing

Run from the repository root:

```bash
python -m pytest Unit-Tests
```

Tests are plain pytest classes (`class TestMain:` with bare `assert`), not `unittest.TestCase`. The current `.vscode/settings.json` enables unittest discovery, which finds 0 tests for that reason; switch VS Code to pytest if you want tests in the Testing panel. Add tests for every new `Library/` module, and make sure the suite passes before committing.

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

  followed by banner sections in this order: `Libraries`, `Files Import`, `FUNCTIONS`.

- Every function has a docstring in this format:

  ```python
  """
  FUNCTION: <name>

  DESCRIPTION: <what it does>

  INPUTS: <args, or None>

  Outputs: <return value, or None>
  """
  ```

- `snake_case` for functions and variables, `PascalCase` for classes, and `if __name__ == "__main__": main()` at the bottom of runnable scripts.
- Top-level folders use `Title-Case-With-Hyphens` (`Image-Data`, `Unit-Tests`).
- Keep README.md in sync when adding commands, the dataset, the model, or results.

## Git

The repo is `aekadan/ecen4273_project2` on GitHub, owned by Aek. Work on a branch and open a pull request rather than pushing to `main`.
