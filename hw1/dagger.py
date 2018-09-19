#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import pickle as pkl
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

def bcnet(outc, obs):
    fc1 = tf.nn.relu(tf.layers.dense(obs, 32))
    fc2 = tf.nn.relu(tf.layers.dense(fc1, 32))
    fc3 = tf.nn.relu(tf.layers.dense(fc2, 32))
    out = tf.layers.dense(fc3, outc)
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    # with tf.Session() as sess:
    #    tf_util.initialize()

    import gym
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit
    obs_ph = tf.placeholder(shape=[None, env.observation_space.shape[0]], dtype=tf.float32)
    output = bcnet(int(env.action_space.shape[0]), obs_ph)
    act_label = tf.placeholder(shape=[None, env.action_space.shape[0]], dtype=tf.float32)
    loss = tf.losses.mean_squared_error(act_label, output)#tf.reduce_mean(tf.square(act_label-output))
    optimizer = tf.train.AdamOptimizer(5e-4).minimize(loss)

    demo = pkl.load(open(os.path.join('expert_data', args.envname + '.pkl'), 'rb'))
    batch_size = 20
    num_epoch = 2000
    demo_obs = demo['observations']
    demo_act = demo['actions']
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        for ep in range(num_epoch):
            epoch_loss = 0
            for i in range(int(50)):
                this_obs = demo_obs[i*batch_size:(i+1)*batch_size,:].squeeze()
                this_act = demo_act[i*batch_size:(i+1)*batch_size,:].squeeze()
                this_loss, _ = sess.run([loss, optimizer], feed_dict={obs_ph: this_obs, act_label: this_act})
                epoch_loss += this_loss
            # print('this epoch ', ep, ' loss is ', epoch_loss)
        # save model as ckpt file
            saver.save(sess, os.path.join('expert_data', 'behavior_clone_'+args.envname+'.ckpt'))
            # test expert policy
            if ep > 10:
                obs = env.reset()
                done = False
                expert_reward = 0
                while not done:
                    action = policy_fn(obs[None,:])
                    obs, r, done, info = env.step(action)
                    expert_reward += r
                if ep == 11:
                    print('expert reward is', expert_reward)
                our_reward = 0
                done = False
                obs = env.reset()
                while not done:
                    obs_dim = env.observation_space.shape[0]
                    obs = obs.reshape(1, obs_dim)
                    action = np.array(sess.run([output], feed_dict={obs_ph: obs})).reshape(env.action_space.shape)
                    obs, r, done, info = env.step(action)
                    our_reward += r
                print('our reward is', our_reward)
        """
        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}
        if os.path.isdir('expert_data') == False:
            os.mkdir('expert_data')
        with open(os.path.join('expert_data', args.envname + '.pkl'), 'wb') as f:
            pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)
        """
if __name__ == '__main__':
    main()
