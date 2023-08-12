#!/usr/bin/env bash

# Activate environment
source .venv/bin/activate

# Run stages
for EX in "ex03R"
do
    # Run workflow
    snakemake --configfile "workflow/configurations/${EX}.yaml" --cores $1

    # Syncronize with remote
    rsync -zaP data/output/models/*ISOMAP* carbonate:~/EMB_dump/models/
    rsync -zaP data/interim/embeddings/*ISOMAP* carbonate:~/EMB_dump/embeddings/

    # Clear space
    rm -f data/interim/embeddings/*ISOMAP*
done
