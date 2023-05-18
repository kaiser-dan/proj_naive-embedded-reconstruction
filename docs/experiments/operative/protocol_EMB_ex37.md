# Embedding Reconstruction - Experiment 37
# Preface
## Contents

- [Embedding Reconstruction - Experiment 37](#embedding-reconstruction---experiment-37)
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
- Project ID: EMB_ex37
- Researchers: Daniel Kaiser
- Version history:

| Version  | Date Designed | Date Conducted | Date Analyzed | Notes          |
| -------- | ------------- | -------------- | ------------- | -------------- |
| **v1.0** | 2023-05-09    |                |     | N2V-PC on LFRs |


## Relevant scripts

The experimental simulations were run through the Python script `workflow/[operative/active]/ex37/EMB_ex37.py`. A dataframe was created within the workflow and the resultant dataframe `results/dataframes/dataframe_EMB_ex37[version]_DK_[date].csv` was treated as the input data to the analysis in `notebooks/viz/analysis_EMB_ex37.ipynb`, a Jupyter notebook.

---

# Summary
## TL;DR



## Experimental Goal
We wish to explore the difference in reconstruction performance of N2V applied with and without care for component-wise embedding settings. A significant difference, especially at large $\theta$ where remnants may consist of many small components, may signify our reconstruction method is confounded by particulars of embeddings across components.



## Hypotheses (if applicable)
We expect to see a lessened dip in performance in LFR reconstruction with N2V at large $\theta$. 


---

# Methods
## Data

We utilize LFR multiplexes with the following parameter settings:

| Parameter           | Value(s)                 | Count |
| ------------------- | ------------------------ | ----- |
| $N$                 | 1000                     | 1     |
| $\mu$               | 0.1                      | 1     |
| $T1\, (\gamma)$     | 2.1                      | 1     |
| $T2$                | 1.0                      | 1     |
| $\langle k \rangle$ | 6                        | 1     |
| max($k$)            | $\sqrt{N} = \sqrt{1000}$ | 1     |
| prob                | 1.0                      | 1     |
| Repetitions         | [1,5]                   | 5    |
|                     |                          |       |
| **Total**           |                          | 5    |

Additionally, for experimental parameters we have the following settings:

| Parameter   | Value(s)                     | Count |
| ----------- | ---------------------------- | ----- |
| $\theta$    | `np.linspace(0.5, 0.95, 11)` | 11    |
| Dimension   | 128                          | 1     |
| Walk length | [30]                  | 1    |
|             |                              |       |
| **Total**   |                              | 11    |

Hence, altogether we have $5*11 = 55$ individual records.

## Procedure

**UPDATE**

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


