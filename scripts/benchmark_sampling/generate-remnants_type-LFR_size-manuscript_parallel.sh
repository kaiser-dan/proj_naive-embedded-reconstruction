#!/usr/bin/env bash
# --- Globals ---
## Fixed paramaters
REPS=$(seq 10)
STRATEGY="RANDOM"

## Specified paramaters
THETAS=$(seq 0.05 0.05 0.95)
FILES=$(find ../../data/input/SYSLFR/edgelists/ -type f -regextype egrep -regex ".*/edgelists_.*N-10000_.*")

# --- Main logic ---
parallel --jobs $1 python observe_remnant.py {1} {2} --strategy "RANDOM" --repeat {3} \; ::: $FILES ::: $THETAS ::: $REPS
# parallel echo python observe_remnants.py {1} {2} --strategy "RANDOM" --repeat {3} \; ::: $FILES ::: $THETAS ::: $REPS
