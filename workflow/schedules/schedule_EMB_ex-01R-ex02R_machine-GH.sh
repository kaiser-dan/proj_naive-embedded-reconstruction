#!/usr/bin/env bash

# Activate environment
source .venv/bin/activate

# Run stages
for EX in "ex01R"
do
    # Run workflow
    snakemake --configfile "workflow/configurations/${EX}.yaml" --cores $1

    # Syncronize with remote
    rsync -zaP data/output/models/*N2V* carbonate:~/EMB_dump/models/
    rsync -zaP data/interim/embeddings/*N2V* carbonate:~/EMB_dump/embeddings/

    # Clear space
    rm -f data/interim/embeddings/*N2V*
done

for EX in "ex02R"
do
    # Run workflow
    snakemake --configfile "workflow/configurations/${EX}.yaml" --cores $1

    # Syncronize with remote
    rsync -zaP data/output/models/*LE* carbonate:~/EMB_dump/models/
    rsync -zaP data/interim/embeddings/*LE* carbonate:~/EMB_dump/embeddings/

    # Clear space
    rm -f data/interim/embeddings/*LE*
done