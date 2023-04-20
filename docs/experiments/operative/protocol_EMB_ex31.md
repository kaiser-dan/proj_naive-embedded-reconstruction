# Embedding Reconstruction - Experiment 31
# Preface
## Contents

- [Embedding Reconstruction - Experiment 31](#embedding-reconstruction---experiment-31)
- [Preface](#preface)
	- [Contents](#contents)
	- [Metadata](#metadata)
	- [Relevant scripts](#relevant-scripts)
- [Summary](#summary)
	- [TL;DR](#tldr)
	- [Experimental Goal](#experimental-goal)
	- [Hypotheses (if applicable)](#hypotheses-if-applicable)
- [Methods](#methods)
	- [Data](#data)
	- [Procedure](#procedure)
- [Results](#results)
- [Analysis](#analysis)
- [Future Work](#future-work)


## Metadata
- Project ID: EMB_ex31
- Researchers: Daniel Kaiser
- Version history:

| Version  | Date Designed | Date Conducted | Date Analyzed | Notes                                                                                                      |
| -------- | ------------- | -------------- | ------------- | ---------------------------------------------------------------------------------------------------------- |
| **v1.0** | 2023-04-11    | 2023-04-12     | 2023-04-12    | Implemented `statsmodels` logistic regressor; swapped feature engineering steps (now aligns before scales) |
| *v1.1*   | 2023-04-12    | 2023-04-12     | 2023-04-13    | Modified scaling so that sum of norms is unity                                                             |
| *v1.2*   | 2023-04-13    | 2023-04-13     | -    | Added simple logging of perfect separability errors |
| v1.2.1   | 2023-04-13    | 2023-04-13     | -    | Expanded error logging to include causes of other NaNs |
| v1.2.2   | 2023-04-13    | 2023-04-13     | -    | Fix in logging file IO |




## Relevant scripts

The experimental simulations were run through the Python script `workflow/[operative/active]/ex31/EMB_ex31.py`. A dataframe was created within the workflow and the resultant dataframe `results/dataframes/dataframe_EMB_ex31[version]_DK_[date].csv` was treated as the input data to the analysis in `notebooks/viz/viz_EMB_ex31.ipynb`, a Jupyter notebook.

---

# Summary
## TL;DR

> Everything hurts and I am dying


## Experimental Goal

Explore various logistic regression-based reconstruction models with a consistent, robust renormalization procedure (embedded vector preprocessing) and logistic regression solver.


## Hypotheses (if applicable)

While the experiment is mostly exploratory, we are generally expecting the performance of the model using embedding and degree features to outperform all others; if one of these features is not relevant, they should have a learned coefficient near 0.

> **Hypothesis 1**: The performance of the reconstruction using embedding distance and configuration degrees (as well as an intercept) should be greater than the performance of either model using either distance xor degrees; this shoudl hold irregardless of system.


---

# Methods
## Data

We utilize four real multiplexes:

- arXiv collaboration network
- _C. Elegans_ connectome
- _Drosophila_ genetic interaction network
- London transportation network

Within these multiplexes, we induce a duplex and restrict our attention therein; respectively, these are:

- physics.data-an, cond-mat.dis-nn (2, 6)
- electric, chem. monadic (1, 2)
- direct, suppressive (1, 2)
- underground, overground (1, 2)

## Procedure
1. [**Set-up**] Fix `system` and `theta`.
2. [**Set-up**] Load cached remnants, observed edge sets, and embedded vectors.
3. [**Pre-processing**] Apply renormalization procedure to embedded vectors.
4. [**Set-up**] Fix `features`.
5. [**Calculations**] Calculate `features` from embedded vectors and/or remnant degree sequences.
6. [**Calculations**] Train logistic regressor on features associated to edges in observed edge.sets.
7. [**Calculations**] Apply logistic regression to edges not in observed edge sets.
8. [**Calculations**] Measure reconstruction's accuracy, R.O.C. A.U.C., and Precision-Recall A.U.C.
9. [**Post-processing**] Add all variables and measurements to a rolling dataframe
10. [**Set-up**] Repeat (4)-(9) for all feature sets.
11. [**Set-up**] Repeat (1)-(10) for all (`system`, `theta`) pairs.
12. [**Post-processing**] Save rolling dataframe to disk.

---

# Results


---

# Analysis


---

# Future Work

We are extending this experiment to a synthetic model in `ex32` - the settings there will emulate previous works (LFRs of varying community strength, layer-wise community correlation, and degree heterogeneity).

Additionally, when more real datasets and cleaned we will apply this procedure on those datasets. Currently, experiment ID is unknown. Simillarly, we will also apply to all possible induced multiplexes of these systems at a later date.