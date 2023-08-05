#!/usr/bin/env bash
# ============= SET-UP ==============
# --- Globals ---
## Pathing
DIR_EDGELISTS="../../data/input/edgelists/"  # Where to find remnants (relative)
FORCEALL=1  # Ignore if cache already exists

## Miscellaneous
BAR_SIZE=20  # Progress bar aesthetics
REGEXTYPE="egrep"  # Regex-type for `find`
CONSOLE_LOG="0"  # Log to console as well as file

# ============= FUNCTIONS ==============
# --- File name processing ---
find_files () {
    # Args (sorted): REGEX_STRING, TEMP_FILE, EMBEDDING
    # ~~~~~~~~~~~~~~~~~~~~
    # Find all elements matching regex string
    find ../../data/input/edgelists/ -type f -regextype egrep -regex "$REGEX_STRING" > "$TEMP_FILE"
    mapfile RAWFILES < "$TEMP_FILE"

    # Declare cleaned array, initially empty
    CLEANFILES=()

    # Check each file and add if they meet all criterion
    for FILE in "${RAWFILES[@]}"
    do
        # Ignore empty filenames
        # ? How do these come up? Can I exclude them from the find command directly?
        [[ ${#FILE} -lt 3 ]] && continue

        # Ignore files that already exist
        if [[ "$FORCEALL" == "1" ]]
        then
            DIM=$(infer_dim $FILE)
            FILE_PATH="${FILE%/*}/"
            FILE_BASE="${FILE##*/}"
            NEWFILE="../../data/input/caches/method-${EMBEDDING}*dim-${DIM}*$FILE_BASE"
            [[ $(check_exists $NEWFILE) == "1" ]] && continue
        fi

        # --- Passed ---
        # Add to array to save
        CLEANFILES+=($FILE)
    done

    # Save cleaned file names to file
    printf "%s\n" "${CLEANFILES[@]}" > "clean_${TEMP_FILE}"
}


infer_dim () {
    # Args (sorted): FILE
    # ~~~~~~~~~~~~~~~~~~~~
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
    # ~~~~~~~~~~~~~~~~~~~~
    [[ -f "$1" ]] && return 1
}

# --- Display ---
update_progbar () {
    # Args (sorted): k, NUM_FILES, BAR_SIZE
    # ~~~~~~~~~~~~~~~~~~~~
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

logger () {
    # Args (sorted): TEXT, LOG_FILE
    # ~~~~~~~~~~~~~~~~~~~~
    if [ "$CONSOLE_LOG" == "1" ]
    then
        echo "$(date +%s) $1" | tee -a $2
    else
        echo "$(date +%s)  $1" >> $2
    fi
}

# --- Cleanup ---
cleanup () {
    find ./ -type f -name "*.tmp" -delete
}

# --- Main logic ---
main () {
    # Args (sorted): REGEX_STRING, EMBEDDING, REPS
    # ~~~~~~~~~~~~~~~~~~~~
    # >>> Book-keeping >>>
    # Args processing
    REGEX_STRING="$1"
    EMBEDDING="$2"
    REPS="$3"
    TEMP_FILE="remnant_filepaths.tmp"
    LOG_FILE=".logs/log_$(date +'%F::%H%M%S').log"

    # Empty log file
    touch $LOG_FILE; rm $LOG_FILE
    # <<< Book-keeping <<<

    # Gather valid remnants for embedding
    find_files $REGEX_STRING $TEMP_FILE $EMBEDDING
    mapfile FILES < "clean_${TEMP_FILE}"

    # Apply main logic
    for ((k = 0; k <= ${#FILES[@]}; k++))
    do
        FILE="${FILES[k]}"

        # Progress bar update
        update_progbar $k ${#FILES[@]} $BAR_SIZE

        # Get DIM from file
        DIM=$(infer_dim $FILE)

        # Cache embedding
        logger ">>>>>>>>>>>>>>" $LOG_FILE
        logger "Embedding $FILE" $LOG_FILE

        logger "Naive" $LOG_FILE
        python embed_and_cache.py $FILE $EMBEDDING $DIM --repeat $REPS

        logger "Per-component" $LOG_FILE
        python embed_and_cache.py $FILE $EMBEDDING $DIM --percomponent --repeat $REPS
        logger "<<<<<<<<<<<<<<" $LOG_FILE
    done
}


# ============= MAIN ==============
# Parse CL arguments
REGEX_STRING=".*/remnants_.*remrep-1.*N-$1_.*"
EMBEDDING="$2"
REPS="$3"

# Embed the remnants
main $REGEX_STRING $EMBEDDING $REPS

# Cleanup
cleanup
