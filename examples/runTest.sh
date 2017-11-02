#! /bin/bash

for i in *.dat; do
  result="${i%.dat}.1"
  echo "M = 1" > $result
  ../linbox/main -F "$i" -M 1 >> $result
  result="${i%.dat}.4"
  echo "M = 4" > $result
  ../linbox/main -F "$i" -M 4 >> $result
  result="${i%.dat}.8"
  echo "M = 8" > $result
  ../linbox/main -F "$i" -M 8 >> $result
done
