#!/usr/bin/env bash

# Activate environment
source .venv/bin/activate

# Run stages
for EX in "ex01" "ex02" "ex03" "ex04"
do
    # Run workflow
    snakemake --configfile "workflow/configurations/${EX}.yaml" --cores $1

    # Syncronize with remote
    rsync -zaP data/output/models/ carbonate:~/EMB_dump/models/
    rsync -zaP data/interim/embeddings/ carbonate:~/EMB_dump/embeddings/

    # Clear space
    rm -f data/interim/embeddings/*
done
