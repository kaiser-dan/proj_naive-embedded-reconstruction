#!/usr/bin/env bash

# Activate environment
source .venv/bin/activate

# Run stages
for EX in "ex04R"
do
    # Run workflow
    snakemake --configfile "workflow/configurations/${EX}.yaml" --cores $1

    # Syncronize with remote
    rsync -zaP data/output/models/*HOPE* carbonate:~/EMB_dump/models/
    rsync -zaP data/interim/embeddings/*HOPE* carbonate:~/EMB_dump/embeddings/

    # Clear space
    rm -f data/interim/embeddings/*HOPE*
done
