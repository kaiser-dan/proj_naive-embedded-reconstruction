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
# --- File name processing ---
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

check_exists () {
    # Args (sorted): FILE
    if [ -f "$1" ]
    then
        return 1
    else
        return 0
    fi
}

# --- Display ---
update_progbar () {
    # Args (sorted): k, NUM_FILES, BAR_SIZE
    _k=$1
    _NUM_FILES=$2
    _BAR_SIZE=$3

    # Re-evaluate normalized task progress
    prog_display=$(echo "scale=0; 100 * $_k / $_NUM_FILES" | bc)  # get current percentage complete
    prog_bar=$(echo "scale=0; $prog_display * $_BAR_SIZE / 100" | bc)  # get current bar percentage in groups of (1/BAR_SIZE)%

    echo -n "[ "  # start bar
    for ((i = 0 ; i <= $prog_bar; i++)); do echo -n "##"; done  # write bar according to progress
    for ((j = i ; j <= $_BAR_SIZE; j++)); do echo -n "   "; done  # write empty bar according to remaining
    echo -n " ] "  # end bar
    echo -n "$prog_display %" $'\r'  # append percentage
}

# --- Cleanup ---
cleanup () {
    find ./ -type f -name "*.tmp" -delete
}

# --- Main logic ---
main () {
    # Args (sorted): REGEX_STRING, TEMP_IDENTIFIER
    # ~~~~~~~~~~~~~~~~~~~~
    # >>> Book-keeping >>>
    # Args processing
    REGEX_STRING="$1"
    TEMP_IDENTIFIER="$2"
    TEMP_FILE="remnants_$TEMP_IDENTIFIER.tmp"
    LOG_FILE=".logs/log_$TEMP_IDENTIFIER.log"
    # <<< Book-keeping <<<

    # Specify remnants to embed
    find ../../data/input/edgelists/ -type f -regextype egrep -regex "$REGEX_STRING" > "$TEMP_FILE"
    mapfile FILES < $TEMP_FILE

    # Apply main logic
    for ((k = 0; k <= ${#FILES[@]}; k++))
    do
        # Check file name exists
        FILE=${FILES[k]}
        check_exists $FILE
        if [[ "$?" == "1" ]]
        then
            echo "$FILE already exists; moving to next file" >> $LOG_FILE
            continue
        fi

        # Progress bar update
        update_progbar $k ${#FILES[@]} $BAR_SIZE

        # Get DIM from file
        DIM=$(infer_dim $FILE)

        # Cache embedding
        echo "Embedding $FILE" >> $LOG_FILE
        echo "Naive" >> $LOG_FILE
        python embed_and_cache.py $FILE $EMBEDDING $DIM --repeat $REPS
        echo "Per-component" >> $LOG_FILE
        python embed_and_cache.py $FILE "$EMBEDDING" $DIM --percomponent --repeat $REPS
        echo "=============" >> $LOG_FILE
    done
}


# ============= MAIN ==============
# Declare what remnants to embed
declare -a REGEXES=(".*/remnants_.*remrep-1.*N-250_.*" ".*/remnants_.*remrep-1.*N-500_.*" ".*/remnants_.*remrep-1.*N-750_.*" ".*/remnants_.*remrep-1.*N-1000_.*")
IDS=$(seq ${#REGEXES[@]})

# Make functions to make visible to GNUParallel
export -f infer_dim
export -f check_exists
export -f update_progbar
export -f main

# Embed the remnants over as many cores as are currently available
parallel --xapply main {1} {2} ::: ${REGEXES[@]} ::: ${IDS[@]}

# Cleanup
cleanup