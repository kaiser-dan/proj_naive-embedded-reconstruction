#!/bin/bash

for FILE in $(ls raw/)
do
    echo "Processing $FILE"
    tail -n +2 raw/$FILE > processed/$FILE
done