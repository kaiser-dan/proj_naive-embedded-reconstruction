#!/usr/bin/env bash

PY="../../.venv/bin/python"

for EMBEDDING in "N2V" "LE" "Isomap" "HOPE"
do
    echo "Profiling $EMBEDDING..."
    for REP in $(seq 9)
    do
        echo "$EMBEDDING - Rep $REP / 9"
        $PY profile_workflow.py $EMBEDDING >> times_$EMBEDDING.data
    done
done
