#!/bin/bash
g++ -std=c++14 parse.cc
./getTests.sh > tests.txt

for i in $(cat tests.txt); do
	echo $i
	./a.out $i >> table.tex
done
