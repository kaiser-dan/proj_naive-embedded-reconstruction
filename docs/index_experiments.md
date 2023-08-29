# Experiment Index

This document briefly summarizes the experimental configurations.

- [Experiment Index](#experiment-index)
- [Global hyperparameters](#global-hyperparameters)
- [Synthetic Multiplex Reconstruction](#synthetic-multiplex-reconstruction)
  - [ex01](#ex01)
  - [ex02](#ex02)
  - [ex03](#ex03)
  - [ex04](#ex04)
  - [ex05](#ex05)
  - [ex06](#ex06)
- [Real multiplex reconstruction](#real-multiplex-reconstruction)
  - [ex07](#ex07)

# Global hyperparameters

All files specified by 'Configuration' are in `workflow/configs/` directory.

All reconstruction experiments utilize relative sizes of trianing sets, $\theta$, in the set:
$
    \\
    \theta \in \{0.05, 0.10, 0.15, 0.20, 0.25,\\
    0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,\\
    0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95\}\\
$
This is because $\theta \in \{0,1\}$ are nonsensical, having either no training data or no task to fulfill.

Additionally, the exponent of the community size distribution is fixed at $t_2 = 1.0$.

# Synthetic Multiplex Reconstruction
## ex01

Reconstruct synthetic LFR model in "easy" case.

- Configuration: `ex01.yaml`
- Notable parameters
  - $t_1 = 2.1$
  - $\mu = 0.1$

## ex02

Reconstruct synthetic LFR models as $\mu$ varies.

- Configuration: `ex02.yaml`
- Notable parameters
  - $t_1 = 2.1$
  - $\mu \in \{0.1, 0.2, 0.3, 0.4, 0.5\}$

## ex03

Reconstruct synthetic LFR with strong community structure and homogeneous degrees.

- Configuration: `ex03.yaml`
- Notable parameters:
  - $t_1 = 4.0$
  - $\mu = 0.1$

## ex04

Reconstruct synthetic LFR with weak community structure and heterogenous degrees.

- Configuration: `ex04.yaml`
- Notable parameters:
  - $t_1 = 2.1$
  - $\mu = 0.5$

## ex05

Reconstruct synthetic LFR with mediocre community structure and midly heterogeneous degrees.

- Configuration: `ex05.yaml`
- Notable parameters:
  - $t_1 = 2.7$
  - $\mu = 0.3$

## ex06

Reconstruct synthetic LFR with imbalanced class sizes.

- Configuration: `ex06.yaml`
- Notable parameters:
  - $t_1 = 2.1$
  - $\mu = 0.1$
- Note that one layer has 1000 nodes and the other has 10,000.

# Real multiplex reconstruction
## ex07

Reconstruct arXiv collaboration multiplex.

