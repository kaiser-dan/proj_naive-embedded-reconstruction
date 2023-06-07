#!/usr/bin/env bash
# --- Globals ---
## Fixed paramaters
REPS=5
STRATEGY="RANDOM"

## Specified paramaters
THETAS=$(seq 0.0 0.05 1.0)
FILES=$(find ../../data/input/edgelists/ -type f -regextype egrep -regex ".*/edgelists_.*N-(250|500|750|1000)_.*")

# --- Main logic ---
# echo "REPS=$REPS"
# echo "THETA space = $THETAS"
for FILE in $FILES
do
    echo "Getting remnants for duplex $FILE"
    for THETA in $THETAS
    do
        echo "Observing with theta = $THETA"
        for REP in $(seq $REPS)
        do
            python observe_remnant.py $FILE $THETA --strategy $STRATEGY --repeat $REPS
        done
    done
done
