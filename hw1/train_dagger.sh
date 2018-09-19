#!/bin/bash
set -eux
#for e in Hopper-v2 Ant-v2 HalfCheetah-v2 Humanoid-v2 Reacher-v2 Walker2d-v2
for e in Ant-v2
do
    python dagger_new.py experts/$e.pkl $e --render --num_rollouts=20
done
