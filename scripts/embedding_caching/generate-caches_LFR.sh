#!/usr/bin/env bash
# --- Globals ---
## Fixed paramaters
REPS=5
EMBEDDING="N2V"

## Specified paramaters
find ../../data/input/edgelists/ -type f -regextype egrep -regex ".*/remnants_.*N-(250|500|750|1000)_.*" > remnants.tmp
mapfile FILES < remnants.tmp

# --- Helpers ---
infer_dim () {
    # N is first argument
    # DIM is output
    if [ "$1" = "250" ]
    then
        echo 32
    elif [ "$1" = "500" ]
    then
        echo 64
    elif [ "$1" = "750" ]
    then
        echo 96
    elif [ "$1" = "1000" ]
    then
        echo 128
    else
        echo "Invalid value of N!"
        return 1
    fi
    return 0
}


# --- Main logic ---
for ((k = 0; k <= ${#FILES[@]}; k++))
do
    # Progress bar update
    # echo -n "[ "  # start bar
    # for ((i = 0 ; i <= k; i++)); do echo -n "###"; done  # write bar according to progress
    # for ((j = i ; j <= ${#FILES[@]}; j++)); do echo -n "   "; done  # write empty bar according to remaining
    # v=$((k * 10))  # format percentage complete
    # echo -n " ] "  # end bar
    # echo -n "$v %" $'\r'  # append percentage

    # Get DIM from N
    N=$(echo ${FILES[k]} | grep -o "N-[0-9]*" | grep -o "[0-9]*")
    DIM=$(infer_dim $N)
    # echo $N $DIM

    # # Embedding native
    python generate-caches_LFR.sh ${FILES[k]} $EMBEDDING $DIM --repeat $REPS

    # # Embedding per-component
    # python generate-caches_LFR.sh ${FILES[k]} $EMBEDDING $DIM --percomponent --repeat $REPS
done


# --- Cleanup ---
# find ./ -type f -name "*.tmp" -delete