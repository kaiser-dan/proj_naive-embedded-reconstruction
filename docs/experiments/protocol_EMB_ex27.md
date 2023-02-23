# Embedding Reconstruction - Experiment 27
# Preface
## Contents

- [Embedding Reconstruction - Experiment 27](#embedding-reconstruction---experiment-27)
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
- Project ID: EMB_ex27
- Researchers: Daniel Kaiser
- Version history:

| Version  | Date Designed | Date Conducted | Date Analyzed | Notes                                                        |
| -------- | ------------- | -------------- | ------------- | ------------------------------------------------------------ |
| *v0.1* | 2023-02-23    | 2022-02-23     | 2022-02-23    | Prototyped refactored workflow |
| *v0.2* | 2023-02-23    | 2022-02-23     | 2022-02-23    | Expanded into "trouble" duplexes, tested error handling |
| **v1.0** | 2023-02-23    | 2022-02-23     | X    | XXX |

## Relevant scripts

The experimental simulations were run through the Python script `workflow/[operative/active]/ex27/EMB_ex27_logreg-degree.py`. A dataframe was created within the workflow and the resultant dataframe `results/dataframes/dataframe_EMB_ex27[version]_DK_[date].csv` was treated as the input data to the analysis in `notebooks/viz/analysis_EMB_ex27.ipynb`, a Jupyter notebook.

---

# Summary
## TL;DR

X

## Experimental Goal

Reproduce Naive Bayes paper's results (under "D" classifier) for real duplexes.


## Hypotheses (if applicable)

We should see quantitatively equivalent behavior to the Naive Bayes classifier paper, reproduced below:

<center><img src="../../results/plots/previous_results.png" alt="previous results"></center>


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
1. [**Set-up**] Load dataset
2. [**Set-up**] Calculate total aggregate $A = \alpha \cup \beta$.
3. [**Set-up**] Observe training set $\Theta = \theta_{\alpha} \cup \theta_{\beta}$ of relative size (per layer) $\theta$.
4. [**Set-up**] Form remnants $\mathcal{R}_{\alpha}, \mathcal{R}_{\beta}$ and aggregate $\tilde{A} = A - \Theta$.
5. [**Feature calculations**] Calculate degrees sequences $k^{\alpha}, k^{\beta}$ of $\mathcal{R}_{\alpha}, \mathcal{R}_{\beta}$.
6. [**Feature calculations**] Calculate configuration degrees
   $$
   \forall e = (i, j) \in \tilde{A} \qquad d_e = \frac{k_i^{\alpha}k_j^{\alpha}}{k_i^{\alpha}k_j^{\alpha} + k_i^{\beta}k_j^{\beta}}
   $$
7. [**Model training**] Train a logistic regression classifier on $\{ d_e \}$.
8. [**Reconstruction**] Reconstruct $\tilde{A}$.
9.  [**Measure performance**] Measure performance using accuracy, AUROC and AUPR.
10. [**Set-up**] Repeat (3) - (9) for some range of $\theta$.

---

# Results


---

# Analysis



---

# Future Work

`EMB_ex28` will expand this analysis to include N2V embedding distance as well.


