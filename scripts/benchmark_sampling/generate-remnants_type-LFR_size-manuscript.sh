#!/usr/bin/env bash
# --- Globals ---
## Fixed paramaters
REPS=10
STRATEGY="RANDOM"

## Specified paramaters
THETAS=$(seq 0.05 0.05 0.95)
FILES=$(find ../../data/input/edgelists/ -type f -regextype egrep -regex ".*/edgelists_.*N-10000_.*")

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
