# Embedding Reconstruction - Experiment 29
# Preface
## Contents

- [Embedding Reconstruction - Experiment 29](#embedding-reconstruction---experiment-29)
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
- Project ID: EMB_ex29
- Researchers: Daniel Kaiser
- Version history:

| Version  | Date Designed | Date Conducted | Date Analyzed | Notes                                                                          |
| -------- | ------------- | -------------- | ------------- | ------------------------------------------------------------------------------ |
| *v0.1*   | 2023-03-21    | 2023-03-22     | 2023-03-22    | Prototyped workflow                                                            |
| **v1.0** | 2023-03-21    | 2023-03-22     | 2023-03-23    | C. Elegans and London                                                          |
| *v1.1*   | 2023-03-21    | 2023-03-22     | 2023-03-23    | arXiv and Drosophila                                                           |
| **v2.0** | 2023-03-23    | XXX            | XXX           | Refactored workflow to use cached network data                                 |
| *v2.1*   |               |                |               |                                                                                |
| *v2.2*   |               |                |               |                                                                                |
| *v2.3*   |               |                |               |                                                                                |
| *v2.4*   |               |                |               |                                                                                |
| *v2.5*   |               |                |               |                                                                                |
| **v3.0** | 2023-03-31    | 2023-04-03     | 2023-04-03    | Added embedding center of mass alignment to source code and ex29 workflow      |
| v3.0.1.  | 2023-04-03    | 2023-04-03     | 2023-04-03    | Attempted hot fix of realigned vector formatting                               |
| v3.0.2   | 2023-04-03    | 2023-04-03     | 2023-04-03    | Fix realigned vector assignment to cache; begin analysis on real networks      |
| **v4.0** | 2023-04-04    | 2023-04-04     | XXX           | Modified alignment to origin, prefaced scaling; exploring both embedding forms |
|          |               |                |               |                                                                                |



## Relevant scripts

The experimental simulations were run through the Python script `workflow/[operative/active]/ex29/EMB_ex29.py`. A dataframe was created within the workflow and the resultant dataframe `results/dataframes/dataframe_EMB_ex29[version]_DK_[date].csv` was treated as the input data to the analysis in `notebooks/viz/analysis_EMB_ex29.ipynb`, a Jupyter notebook.

---

# Summary
## TL;DR



## Experimental Goal




## Hypotheses (if applicable)


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
5. !!! UPDATE PROCEDURE!!!
6. [**Reconstruction**] Reconstruct $\tilde{A}$.
7.  [**Measure performance**] Measure performance using accuracy, AUROC and AUPR.
8.  [**Set-up**] Repeat (3) - (9) for some range of $\theta$.

---

# Results


---

# Analysis


---

# Future Work


