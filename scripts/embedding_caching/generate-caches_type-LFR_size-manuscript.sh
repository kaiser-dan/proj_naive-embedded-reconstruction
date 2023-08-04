#!/usr/bin/env bash
# Clarify arguments
## Globals
DIM=128
INPUT_DIR="../../data/input/SYSLFR/edgelists"

## Command-line
EMBEDDING="$1"

# Get remnant filehandles
find $INPUT_DIR -type f -regextype egrep -regex ".*theta-0.[4-9][0-9].*remrep-0.*mu-0.[2-5].*" > "filehandles_$1.tmp"

# Apply embed_and_cache.py
cat "filehandles_$1.tmp" | parallel --jobs $2 python embed_and_cache.py {} $EMBEDDING $DIM

# Remove temporary files
rm "filehandles_$1".tmp
