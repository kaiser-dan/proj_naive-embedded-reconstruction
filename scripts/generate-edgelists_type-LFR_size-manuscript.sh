#!/usr/bin/env bash
# --- Globals ---
N=10000
T1=2.1
T2=1.0
KAVG=20
PROB=1.0
KMAX=100


# --- Main logic ---
for MU in 0.1 0.2 0.3 0.4 0.5
do
    # * This cannot be parallelized naively
    # * LFR binary saves to file without uniquely identifying name
    # * Piping into GNU parallel would cause a data race
    echo "Sampling duplexes with Mu=${MU}..."
    python sample_LFR.py -N $N -u $MU -d $T1 -c $T2 -k $KAVG -m $KMAX -p $PROB
done

# Cleanup LFR binary temporary files
find ./ -type f -regextype egrep -regex ".*\.dat" -delete
