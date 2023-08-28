# Experiment Index

This document briefly summarizes the experimental configurations.

- [Experiment Index](#experiment-index)
  - [Global hyperparameters](#global-hyperparameters)
  - [ex01](#ex01)
  - [ex02](#ex02)

## Global hyperparameters

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

