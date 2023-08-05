# Embedding Reconstruction - Experiment 32
# Preface
## Contents

- [Embedding Reconstruction - Experiment 32](#embedding-reconstruction---experiment-32)
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
- Project ID: EMB_ex32
- Researchers: Daniel Kaiser
- Version history:

| Version  | Date Designed | Date Conducted | Date Analyzed | Notes                                                   |
| -------- | ------------- | -------------- | ------------- | ------------------------------------------------------- |
| **v1.0** | 2023-04-13    | 2023-04-13     | 2023-04-13    | Preliminary experiment with $\mu = 0.1$, _prob_ varies  |
| *v1.1*   | 2023-04-16    | 2023-04-16     | 2023-04-16    | Full run with $\mu, prob$ varying.                      |
| **v2.0** | 2023-04-16    | 2023-04-16     | 2023-04-17    | Full run with `scikit-learn` package and regularization |
| *v2.1*   | 2023-04-17    | 2023-04-17     | 2023-04-17    | Rescaled distances with MinMax procedure |


## Relevant scripts

The experimental simulations were run through the Python script `workflow/[operative/active]/ex32/EMB_ex32.py`. A dataframe was created within the workflow and the resultant dataframe `results/dataframes/dataframe_EMB_ex32[version]_DK_[date].csv` was treated as the input data to the analysis in `notebooks/viz/analysis_EMB_ex32.ipynb`, a Jupyter notebook.

---

# Summary
## TL;DR



## Experimental Goal




## Hypotheses (if applicable)


---

# Methods
## Data

We utilize LFR multiplexes with the following parameter settings:

**FILL IN EVENTUALLLY**

## Procedure
1. [**Set-up**] Fix `mu`, `prob`, and `theta`.
2. [**Set-up**] Load cached remnants, observed edge sets, and embedded vectors.
3. [**Pre-processing**] Apply renormalization procedure to embedded vectors.
4. [**Set-up**] Fix `features`.
5. [**Calculations**] Calculate `features` from embedded vectors and/or remnant degree sequences.
6. [**Calculations**] Train logistic regressor on features associated to edges in observed edge.sets.
7. [**Calculations**] Apply logistic regression to edges not in observed edge sets.
8. [**Calculations**] Measure reconstruction's accuracy, R.O.C. A.U.C., and Precision-Recall A.U.C.
9. [**Post-processing**] Add all variables and measurements to a rolling dataframe
10. [**Set-up**] Repeat (4)-(9) for all feature sets.
11. [**Set-up**] Repeat (1)-(10) for all (`mu`, `prob`, `theta`) combinations.
12. [**Post-processing**] Save rolling dataframe to disk.

---

# Results


---

# Analysis


---

# Future Work


