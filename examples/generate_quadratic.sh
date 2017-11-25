#!/bin/bash

for n in {1..15}
do
    magma -b n:=$n random-quadratic.mgm solve.mgm >> all.dat
done
