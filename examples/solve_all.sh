#! /bin/bash

rm -f all.dat
for i in magma_inputs/*.mgm; do
    magma -b $i solve.mgm >> all.dat
done
