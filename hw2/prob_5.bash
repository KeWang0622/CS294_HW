#!/bin/bash
set -eux
for e in 1000 2000 5000 10000 20000
do
    for lr in 0.001 0.005 0.01 0.02 0.05
    do
        echo $e
        echo $lr
        python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -b $e -lr $lr -rtg --exp_name hc_b_$e._r_$lr
    done
done

