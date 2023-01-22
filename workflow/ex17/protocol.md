# Metadata

- Project ID: EMB_ex17
- Researchers: DK
- Version History:

| Version | Date Started | Date Finished | Notes                                          |
| ------- | ------------ | ------------- | ---------------------------------------------- |
| v1.0    | 2023-01-21   | XXX           | Refactored workflow; not backwards compatible! |
|         |              |               |                                                |


# Experiment
## Goal

Sweep N2V walk length parameter in real multiplex embedding reconstruction.

## Procedure
### Theoretical procedure

1. Do stuff

### Snakemake procedure

The Snakemake workflow follows the above procedure. The rules from this experiment's Snakefile are available ordered below; additionally, embedded below is the workflow rule graph, once developed.

**Experiment**

1. `observe_remnants` - If remnants are not available on disk (`input/preprocessed` directory), then observe some partial information and save the remnants to disk.
2. `embed_X` - If embedded vectors are not available on disk (`output/raw` directory), then embed with specified parameters and save vectors to disk.
3. `calculate_distances` - If distances are not available on disk (`output/raw` directory), then calculate distances with specified parameters and save to disk.
4. `reconstruct_multiplex` - If reconstructed layers are not available on disk (`output/processed` directory), then reconstruct layers with specified parameters and save to disk.

**Analysis**

5. `measure_performance` - If performance of reconstruction is not available on disk (`output/processed` directory), then measure the reconstruction performance and save record to disk.
6. `aggregate_records` - Gather all reconstruction performance records and concatenate into a single dataframe. Save to `results/dataframes` directory.
7. `basic_figures` - Using the dataframe from `results/dataframes` directory, create some figures of basic relationships (e.g. performance versus some parameters). Save figures to `results/figures` directory.

**Documentation**

8. `save_rulegraph` - Create a rulegraph exemplifying the rule call D.A.G. for this experiment. Save figure to `results/reports` directory.
9. `generate_report` - _NOT IMPLEMENTED_.
10. `all` - Gather all relevant output files from the experiment. Note: Controls Snakemake D.A.G. navigation, do not edit!

## Notes