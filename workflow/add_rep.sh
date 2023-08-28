#!/usr/bin/env bash

for file in $(ls targets/)
do
    # TODO: Fix erroneous TARGET wildcard replacement
    sed -i 's/theta\}_/theta\}_rep-{rep}_/' $file
    sed -i 's/rep=[1,2,3,4,5], layerpair/rep=\[1,2,3,4,5\], layerpair/' $file
    sed -i 's/rep=[1,2,3,4,5], mplx/rep=\[1,2,3,4,5\], mplx/' $file
done
