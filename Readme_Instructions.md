# CS294-112 HW 1: Imitation Learning

Instructions:
I implemented imitation learning using TensorFlow
1. Run Behavior Leaning on Hopper:
python bc_new.py experts/Hopper-v2.pkl Hopper-v2 --render --num_rollouts=20

2. Run Behavior Leaning on Ant:
python bc_new.py experts/Ant-v2.pkl Ant-v2 --render --num_rollouts=20

3. Run Bagger on Hopper:
python dagger_new.py experts/Hopper-v2.pkl Hopper-v2 --render --num_rollouts=20

4. Run Dagger on Ant:
python dagger_new.py experts/Ant-v2.pkl Ant-v2 --render --num_rollouts=20

5. Run Nonlinearity Behavior Cloning on Ant:
python by_nonlinear.py experts/Ant-v2.pkl Ant-v2 --render --num_rollouts=20

6. Run Nonlinearity Dagger on Ant:
python dagger_nonlinear.py experts/Ant-v2.pkl Ant-v2 --render --num_rollouts=20

Figures:

Run each python file in the folder : "Figures's_scripts"
Remeber to change the directory in the scripts to your csv files' directory.

All the results have been saved to csv files in different tasks. 
If there is any question, please feel free to contact:
kewang@berkeley.edu
510-717-2737