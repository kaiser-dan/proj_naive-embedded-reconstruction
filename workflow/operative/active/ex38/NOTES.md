# Experiment Workflow TDD

## Goal

Reconstruct multiplex from partial observations

## Data

Workflow should be blind to real/synthetic datasets, despite usual caching methodology.

So input to script should be pickled cache?
> **Note:** Caches are, as currently formatted, separated based on $\theta$ as well as network parameters and embedding method!

## Workflow

**TDD**
1. Fix point in parameter grid.
2. Bring corresponding cache into memory.
3. Train classifier
4. Reconstruct multiplex
5. Measure performance
6. Save records

**Some thoughts**
- 1-2 could be a setup function
- 3 is a small function
- 4 is a small function
- 5 is a small function
- 6 is a small post-processing function
  - May have data race problems with snakemake
  - Should cache some temporary files and add a cleanup step?
- 2 uses CachedEmbedding class
- 3 uses LogReg(Model) class
- I will likely need a separate workflow for doing the embeddings themselves
