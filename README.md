# Neural Network Models and Applications Final Exam

Course final project for **神经网络的模型与应用**.

- Student: 滕子珺
- Student ID: 25110180046
- Repository: <https://github.com/Zijun-Teng/neural-network-final-exam>
- Selected problems: **2, 3, 4, 5, 7**

This repository contains the reproducible code, small input data, generated figures, numerical result files, and separated audio files used in the final report.

## Project Structure

```text
.
├── data/
│   ├── BSS/
│   │   ├── 110000001mix1.wav
│   │   ├── 110000001mix2.wav
│   │   └── 110000001mix3.wav
│   └── maze/
│       └── maze.jpg
├── outputs/
│   ├── bss/
│   ├── direction/
│   ├── ei/
│   └── maze/
├── scripts/
│   ├── bss_experiment.py
│   ├── direction_estimation.py
│   ├── ei_hopf.py
│   └── maze_rl.py
├── requirements.txt
└── README.md
```

## Problem Mapping

| Problem | Topic | Script / Output |
| --- | --- | --- |
| 2 | Ordinary renewal process statistics | Analytical derivation in the report |
| 3 | Hopf bifurcation in an E-I rate model | `scripts/ei_hopf.py`, `outputs/ei/` |
| 4 | Poisson population coding and direction estimation | `scripts/direction_estimation.py`, `outputs/direction/` |
| 5 | Blind source separation for audio mixtures | `scripts/bss_experiment.py`, `outputs/bss/` |
| 7 | Maze shortest path via reinforcement learning / Bellman optimality | `scripts/maze_rl.py`, `outputs/maze/` |

## Environment

The code is intentionally lightweight and uses only common scientific Python packages.

```bash
python -m pip install -r requirements.txt
```

Tested with Python 3.12.

Required packages:

- `numpy`
- `scipy`
- `pillow`
- `opencv-python`

## Reproduce All Experiments

Run the following commands from the repository root:

```bash
python scripts/ei_hopf.py
python scripts/direction_estimation.py
python scripts/bss_experiment.py
python scripts/maze_rl.py
```

Each script writes figures and numerical summaries under `outputs/`.

## Push to GitHub

This directory is already initialized as a local Git repository. If the remote repository has not been pushed yet, run:

```bash
git remote set-url origin git@github.com:Zijun-Teng/neural-network-final-exam.git
git push -u origin main
```

GitHub may ask for authentication. For SSH push, make sure the public key on this machine has been added to the GitHub account or has access to this repository.

## Main Numerical Results

### Problem 3: E-I Hopf Bifurcation

Parameters:

```text
tauE=1, MEE=2, MEI=4, MIE=1, MII=1, hE=0.5, hI=0
```

The Hopf condition gives the critical value:

```text
tauI* = 2
```

The maximum real part of the eigenvalues crosses zero at this value. Figures are saved as:

- `outputs/ei/ei_bifurcation.png`
- `outputs/ei/ei_phase.png`

### Problem 4: Direction Estimation

Simulation setup:

```text
N = 61 neurons
preferred directions equally spaced in [0, 180]
sigma = 18
observation window = 25
true direction = 72
Monte Carlo trials = 2000
```

Result:

```text
bias = 0.027500
variance = 0.925416
MSE = 0.925710
Fisher information = 1.160005
CRLB = 0.862065
```

The empirical MSE is close to the Cramér-Rao lower bound, consistent with the asymptotic efficiency of the MLE.

### Problem 5: Blind Source Separation

Compared methods:

- PCA whitening
- FastICA

No ground-truth sources are provided, so no-reference quality indicators are used:

| Method | Mean absolute off-diagonal correlation | Mean absolute kurtosis |
| --- | ---: | ---: |
| Mixtures | 0.001717 | 0.348715 |
| PCA whitening | 0.000000 | 0.547114 |
| FastICA | 0.000000 | 1.130282 |

FastICA preserves decorrelation while increasing non-Gaussianity, giving a better separation according to ICA assumptions.

Generated files:

- `outputs/bss/waveforms.png`
- `outputs/bss/fastica_source_1.wav`
- `outputs/bss/fastica_source_2.wav`
- `outputs/bss/fastica_source_3.wav`

### Problem 7: Maze Planning

The maze image is converted to a deterministic grid-world MDP:

- black pixels: free states
- white pixels: walls
- yellow dot: start state
- actions: up, down, left, right

Result:

```text
grid size = 520 x 392
free states = 159283
start = (507, 354)
goal = (264, 193)
BFS shortest path length = 5730
Bellman policy path length = 5730
```

The Bellman optimal policy reaches the same length as the independent BFS shortest-path check.

Generated file:

- `outputs/maze/maze_path.png`

## Notes

- The BSS task has no reference source signals, so the evaluation uses output cross-correlation and kurtosis instead of SDR/SIR.
- The maze solver first checks reachability. If the specified goal is not connected to the start state, it reports that the target is unreachable.
- ICA output is identifiable only up to source permutation, sign, and scale; this is expected and does not affect the separation objective.
