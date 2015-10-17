#!/bin/sh

for i in $(seq 0 44)
do
    echo "forking process $i"
    python kl_exp.py $i > "$i.txt" 2>&1 &
done

