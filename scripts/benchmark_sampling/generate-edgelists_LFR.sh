#!/usr/bin/env bash
# --- Globals ---
MU=0.1
T1=2.1
T2=1.0
KAVG=10
PROB=1.0
NS=(250 500 750 1000)
KMAXS=(16 22 27 32)

# --- Main logic ---
for index in $(seq 0 3)
do
    N=${NS[index]}
    KMAX=${KMAXS[index]}

    echo "Sampling duplexes with N=${N} & max(k)=${KMAX}..."
    python sample_LFR.py -N $N -u $MU -d $T1 -c $T2 -k $KAVG -m $KMAX -p $PROB
done

# Cleanup LFR binary temporary files
find ./ -type f -regextype egrep -regex ".*\.dat" -delete
