# Assignment 1

Computer vision experiments for feature detection, description, matching,
homography estimation, image stitching, and evaluation.

## Project structure

```text
Assignment1/
├── data/                 # Input images and datasets
├── results/
│   ├── figures/          # Generated plots and visualizations
│   └── tables/           # Generated metrics and experiment tables
├── src/                  # Source code
└── requirements.txt      # Python dependencies
```

## Setup

Use Python 3.10 or newer:

```bash
python -m venv .venv
```

Activate the environment and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the project

The main entry point is `src/main.py`:

```bash
python src/main.py
```

Add input data under `data/` and write generated figures and tables to the
corresponding subdirectories under `results/`.
