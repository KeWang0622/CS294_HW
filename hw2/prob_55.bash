#!/bin/bash
set -eux
for e in 500
do
    for lr in 0.05 0.1 0.2 0.5 1 5 10
    do
        echo $e
        echo $lr
        python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -b $e -lr $lr -rtg --exp_name hc_b_$e._r_$lr
    done
done

