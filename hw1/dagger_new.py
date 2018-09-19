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
import csv


def bcnet(outc, obs):
    fc1 = tf.nn.relu(tf.layers.dense(obs, 64))
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

    
    batch_size = 20
    num_epochs = [1,5,10,30,50,100,200,500,1000]


    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        for num_epoch in num_epochs:
            print(num_epoch)
            sess.run(init_op)
            demo = pkl.load(open(os.path.join('expert_data', args.envname + '.pkl'), 'rb'))
            demo_obs = demo['observations']
            demo_act = demo['actions']
            for ep in range(num_epoch):
                print(ep)
                epoch_loss = 0
                for i in range(int(demo_obs.shape[0]/batch_size)):
                    this_obs = demo_obs[i*batch_size:(i+1)*batch_size,:].squeeze()
                    this_act = demo_act[i*batch_size:(i+1)*batch_size,:].squeeze()
                    this_loss, _ = sess.run([loss, optimizer], feed_dict={obs_ph: this_obs, act_label: this_act})
                    epoch_loss += this_loss
                # print('this epoch ', ep, ' loss is ', epoch_loss)
            # save model as ckpt file
                #saver.save(sess, os.path.join('expert_data', 'behavior_clone_'+args.envname+'.ckpt'))
                # test expert policy
                if ep % 100 ==0:
                    print('The epochs is ',ep)
                obs_dagger = []
                act_dagger = []
                ret_dagger = []

                returns_dagger = 0
                done = False
                obs = env.reset()
                while not done:
                    obs_dagger.append(obs)
                    
                    obs_dim = env.observation_space.shape[0]
                    action_d = policy_fn(obs[None,:])
                    act_dagger.append(action_d)
                    obs = obs.reshape(1, obs_dim)
                    action = np.array(sess.run([output], feed_dict={obs_ph: obs})).reshape(env.action_space.shape)                    
                    obs, r, done, info = env.step(action)
                    returns_dagger +=r
                #print('Our dagger reward is: ',returns_dagger)
                demo_obs = np.vstack((demo_obs,np.array(obs_dagger)))
                #print(demo_act.shape)
                dim = np.array(act_dagger).shape
                demo_act1 = np.vstack((demo_act.reshape((demo_act.shape[0],demo_act.shape[2])),np.array(act_dagger).reshape((dim[0],dim[2]))))
                
                demo_act = demo_act1.reshape((demo_act1.shape[0],demo_act.shape[1],demo_act.shape[2]))

     
            # run expert policy and calculate the mean and std
            returns_exp = []
            for ii in range(args.num_rollouts):
                print('iter', ii)
                exp_reward = 0
                done = False
                obs = env.reset()    
                while not done:
                    action = policy_fn(obs[None,:])
                    obs, r, done, info = env.step(action)
                    exp_reward += r
                    #if args.render:
                       # env.render()
                    #print('Env Rendering when Ep = ',ep)
                print('expert returns', exp_reward)
                returns_exp.append(exp_reward)
            csvFile = open('daggere_'+args.envname+'_'+str(num_epoch)+'_expert.csv','w', newline='')
            writer = csv.writer(csvFile)    
            
            print('experts returns', returns_exp)
            print('experts mean return', np.mean(returns_exp))
            print('experts std of return', np.std(returns_exp))
            returns_exp.append(np.mean(returns_exp))
            returns_exp.append(np.std(returns_exp))
            writer.writerow(returns_exp)


            # run our behavior cloning method and calculate mean and std
            returns_ours = []
            for ii in range(args.num_rollouts):
                print('iter', ii)
                our_reward = 0
                done = False
                obs = env.reset()    
                while not done:
                    obs_dim = env.observation_space.shape[0]
                    obs = obs.reshape(1, obs_dim)
                    action = np.array(sess.run([output], feed_dict={obs_ph: obs})).reshape(env.action_space.shape)
                    obs, r, done, info = env.step(action)
                    our_reward += r
                    if ii == 11:
                        if args.render:
                            env.render()
                    #print('Env Rendering when Ep = ',ep)
                print('our reward is', our_reward)
                returns_ours.append(our_reward)
            csvFile1 = open('dagger_'+args.envname+'_'+str(num_epoch)+'_expert.csv','w', newline='')
            writer1 = csv.writer(csvFile1)   
            print('our returns', returns_ours)
            print('our mean return', np.mean(returns_ours))
            print('our std of return', np.std(returns_ours))             
            returns_ours.append(np.mean(returns_ours))
            returns_ours.append(np.std(returns_ours))
            writer1.writerow(returns_ours)


            csvFile.close()
            csvFile1.close()
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
