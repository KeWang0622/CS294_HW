# CS294-112 HW 1: Imitation Learning

Dependencies:
 * Python **3.5**
 * Numpy
 * TensorFlow
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.10.5**

Once Python **3.5** is installed, you can install the remaining dependencies using `pip install -r requirements.txt`.

**Note**: MuJoCo versions until 1.5 do not support NVMe disks therefore won't be compatible with recent Mac machines.
There is a request for OpenAI to support it that can be followed [here](https://github.com/openai/gym/issues/638).

**Note**: Students enrolled in the course will receive an email with their MuJoCo activation key. Please do **not** share this key.

The only file that you need to look at is `run_expert.py`, which is code to load up an expert policy, run a specified number of roll-outs, and save out data.

In `experts/`, the provided expert policies are:
* Ant-v2.pkl
* HalfCheetah-v2.pkl
* Hopper-v2.pkl
* Humanoid-v2.pkl
* Reacher-v2.pkl
* Walker2d-v2.pkl

The name of the pickle file corresponds to the name of the gym environment.

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