#!/usr/bin/env bash
# Clarify arguments
## Globals
DIM=128
INPUT_DIR="../../data/input/SYSREAL/edgelists"

## Command-line
EMBEDDING="$1"

# Get remnant filehandles
find $INPUT_DIR -type f -regextype egrep -regex ".*remrep-0.*" > filehandles.tmp

# Apply embed_and_cache.py
cat filehandles.tmp | parallel --jobs 10 python embed_and_cache.py {} $EMBEDDING $DIM

# Remove temporary files
rm filehandles.tmp
