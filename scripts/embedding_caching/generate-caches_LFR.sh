#!/usr/bin/env bash
# ============= SET-UP ==============
# --- Globals ---
## Fixed paramaters
REPS=1
EMBEDDING="N2V"

## Miscellaneous
BAR_SIZE=20
touch log.log; rm log.log

# ============= FUNCTIONS ==============
# --- Helpers ---
infer_dim () {
    # Args (sorted): FILE
    N=$(echo $1 | grep -o "N-[0-9]*" | grep -o "[0-9]*")

    if [ "$N" = "250" ]
    then
        echo 32
    elif [ "$N" = "500" ]
    then
        echo 64
    elif [ "$N" = "750" ]
    then
        echo 96
    elif [ "$N" = "1000" ]
    then
        echo 128
    else
        echo "Invalid value of N!"
        return 1
    fi
    return 0
}

update_progbar () {
    # Args (sorted): k, NUM_FILES, BAR_SIZE
    _k=$1
    _NUM_FILES=$2
    _BAR_SIZE=$3

    # Re-evaluate normalized task progress
    prog_display=$(echo "scale=0; 100 * $_k / $_NUM_FILES" | bc)  # get current percentage complete
    prog_bar=$(echo "scale=0; $prog_display / $_BAR_SIZE" | bc)  # get current bar percentage in groups of (1/BAR_SIZE)%

    echo -n "[ "  # start bar
    for ((i = 1 ; i <= $prog_bar; i++)); do echo -n "###"; done  # write bar according to progress
    for ((j = i ; j <= $_BAR_SIZE; j++)); do echo -n "   "; done  # write empty bar according to remaining
    echo -n " ] "  # end bar
    echo -n "$prog_display %" $'\r'  # append percentage
}

# --- Cleanup ---
cleanup () {
    find ./ -type f -name "*.tmp" -delete
}


# ============= MAIN ==============
# Specify remnants to embed
# find ../../data/input/edgelists/ -type f -regextype egrep -regex ".*/remnants_.*N-(250|500|750|1000)_.*" > remnants.tmp
find ../../data/input/edgelists/ -type f -regextype egrep -regex ".*/remnants_.*N-250_.*" > remnants.tmp
mapfile FILES < remnants.tmp

# Apply main logic
for ((k = 0; k <= ${#FILES[@]}; k++))
do
    FILE=${FILES[k]}

    # Progress bar update
    update_progbar $k ${#FILES[@]} $BAR_SIZE

    # Get DIM from file
    DIM=$(infer_dim $FILE)

    # Cache embedding
    echo "Embedding $FILE" >> log.log
    echo "Naive" >> log.log
    python embed_and_cache.py $FILE $EMBEDDING $DIM --repeat $REPS
    echo "Per-component" >> log.log
    python embed_and_cache.py $FILE "$EMBEDDING" $DIM --percomponent --repeat $REPS
done

# Cleanup
cleanup