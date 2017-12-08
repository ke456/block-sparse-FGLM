#!/bin/bash

exe="sfglm-eigen"

for i in $(ls ../examples/data/|egrep "dat$"); do
	output="results/${i%.dat}"
	echo "output $output"
	input="../examples/data/$i"
	echo $i
	./$exe -F $input -M 1 -t 32 > "${output}1.out"
	./$exe -F $input -M 2 -t 32 > "${output}2.out"
	./$exe -F $input -M 4 -t 32 > "${output}4.out"
done
