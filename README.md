# Neural Network Models and Applications Final Exam

This repository contains reproducible code and small data files for the final exam report of **神经网络的模型与应用**.

Student: 滕子珺  
Student ID: 25110180046

## Contents

- `scripts/ei_hopf.py`: Hopf bifurcation verification for the excitatory-inhibitory rate model.
- `scripts/direction_estimation.py`: Poisson population coding simulation and MLE/CRLB comparison.
- `scripts/bss_experiment.py`: blind source separation using PCA whitening and FastICA.
- `scripts/maze_rl.py`: deterministic grid-world MDP solution for the maze path planning problem.
- `data/`: audio mixtures and maze image used by the scripts.
- `outputs/`: generated figures, result tables, and separated audio files.

## Environment

The scripts use only lightweight scientific Python packages:

```bash
python -m pip install -r requirements.txt
```

Tested with Python 3.12.

## Reproduce Results

Run from the repository root:

```bash
python scripts/ei_hopf.py
python scripts/direction_estimation.py
python scripts/bss_experiment.py
python scripts/maze_rl.py
```

Each script writes its outputs under `outputs/`.

## Notes

The maze script treats black pixels as free states and white pixels as walls. The Bellman optimality solution is equivalent to shortest-path planning in this deterministic MDP with unit step cost; BFS is used as an independent check of the reported path length.

The BSS task has no ground-truth sources in the provided attachment, so the report uses no-reference quality indicators: output cross-correlation and marginal kurtosis.
