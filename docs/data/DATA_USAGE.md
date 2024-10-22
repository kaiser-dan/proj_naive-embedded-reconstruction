# Data Usage Guide

This document will document the data used and generated by pipelines in this repository.
Included in this document are items such as, but not limited to,
- Data directory structure
- File naming convention
- Using the index helper scripts
- Getting metadata


This document was originally created by [Daniel Kaiser](mailto:kaiserd@iu.edu) on 2023-04-27. Updates to this document are not tracked explicitly.

---

# Table of Contents

- [Data Usage Guide](#data-usage-guide)
- [Table of Contents](#table-of-contents)
- [The Data Directory](#the-data-directory)
  - [Tree structure](#tree-structure)
  - [Naming conventions](#naming-conventions)
  - [Large files](#large-files)
- [Data Index and Metadata](#data-index-and-metadata)
  - [Index](#index)
  - [Metadata](#metadata)
  - [More on naming conventions](#more-on-naming-conventions)
  - [Searching data](#searching-data)
- [Helper Scripts](#helper-scripts)
  - [Search and retrieval](#search-and-retrieval)
  - [Creating your own scripts](#creating-your-own-scripts)
- [SCRATCH NOTES](#scratch-notes)


# The Data Directory
## Tree structure
If you have the `tree` bash utility installed, you can see the directory structure of the `data/` directory by running the following command inside a terminal
```bash
tree -d data/
```

Regardless of this utility, broadly speaking we organize the data as follows:

1. At the root of `data/` are this document, data dictionaries, and helper utility for indexing datasets.
2. The first level of subdirectories includes `input/` and `output/`, where we keep the input multiplex systems for experiments and the output reconstructions and experimental observations.
<!-- 3. In both of these directories, we split into a `raw/` and a `[pre]processed/` directory, where original edgelists/reconstructions are kept and any cleaned caches/duplexes/dataframes of experimental observations can be found. -->

## Naming conventions

Since we provide utility to search for data matching with certain characteristics/attributes, we elect to enforce a relatively strict naming convention on individual datafiles. Broadly, all data files associated to our experiments or generated by our source code will follow the below convention:

`<data type>_[<attribute name>-<attribute value>_...]_<date>.<extension>`

This allows for quick searches based on data type, attributes, or date synthesized. Not all attributes are necessarily listed in every file, nor is date always present if there is little reason to expect change in the data or temporal comparison are otherwise unneeded (e.g., in downloaded real multiplexes from other repositories). Some examples of this are listed below (not necessarily present in directory):

- `multiplex_system-celegans_layers-1-2_weighted-False.edgelist`
- `cache_system-LFR_embedding-LE_PC-False_dimensions-128_N-1000_mu-0.1_t1-2.1.pkl`
- `dataframe_EMB_ex37v1.0_20230516.csv`
- `plot_EMB_ex37v2.3_y-auroc_x-pfi_style-line_hue-dimensions_col-walklength_20230521.png`

## Large files

HMMMM

# Data Index and Metadata
## Index

## Metadata

There are many reasons to suspect that metadata is 

## More on naming conventions

## Searching data

# Helper Scripts
## Search and retrieval

## Creating your own scripts

# SCRATCH NOTES
Data hierarchy includes

Network edgelist
    -> Remnants [params: theta, hyperparams: observation_strategy]
        -> Embeddings [params: method, dimension]
            -> CachedEmbeddings
                -> Reconstruction
                    -> Dataframe record
        -> Dataframe
    -> (merged dataframes)
-> (merged dataframes)
