#!/bin/bash

for x in $(cat list.txt)
do
    cd $x
    #rm *py; rm *dat; rm dftb*; rm test*
    #rm temp; rm sub.pbs; rm \#calc.log; rm old* 
    python ../solvation_entropy.py input.txt > tests.dat
    cat tests.dat | tail -n 1
    cd ..
done
    
