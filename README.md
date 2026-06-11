# Quantum Feature Map Evaluation – Minimum Accuracy with Monte Carlo Axis Selection

This repository contains the Python implementation of the generalized minimum accuracy framework for quantum kernel methods, as described in the paper *"Scalable Certified Lower Bounds for Quantum Feature Maps via Monte Carlo Axis Selection"*.

## Main File

- **`classi.py`** – The main script that runs the full experiment (evaluation of quantum feature maps, deterministic and Monte Carlo `R_min` estimation, SVM baselines, and generation of figures and result tables).

## Basic Configurations

All experimental settings are defined at the top of `classi.py` as global variables. The most important ones are:

| Variable | Description | Typical Values |
|----------|-------------|----------------|
| `N_QUBITS` | Number of qubits (feature space dimension = 4^n) | 6 (default) |
| `N_TRAIN` | Number of training samples used | 200 |
| `N_REPETITIONS` | Number of independent runs (repetitions) | 30 |
| `RUN_DETERMINISTIC` | Whether to evaluate all Pauli axes (exact `R_min`) | `True` for n ≤ 6, `False` for larger n |
| `MAX_D_FOR_PAULI_SVM` | Max dimension to train the exact linear SVM in Pauli space | 4096 (i.e., n ≤ 6) |
| `FEATURE_MAP_TYPE` | Type of quantum feature map | `"RY_CRZ"`, `"ZZ"`, or `"RANDOM"` |
| `N_LAYERS` | Number of repetitions of the feature map block | 2 |
| `SCALE_FACTOR` | Scaling factor for rotation angles | `np.pi` |
| `INCLUDE_CLASSICAL_BASELINES` | Include SVM on original features for reference | `True` or `False` |

## Quick Start

1. Install dependencies:
   ```bash
   pip install qiskit scikit-learn numpy pandas matplotlib seaborn
