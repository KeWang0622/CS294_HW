#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 22:33:02 2018

@author: kewang
"""
#Data = open('/Users/kewang/HW3_Figures/ddqn/results/PongKWANG97500001538582828.768958_data.pkl','rb')
#%%
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import math
import pickle as pkl
#Data_d = open('/Users/kewang/HW3_Figures/ddqn/results/PongKWANG97500001538582828.768958_data.pkl','rb')
Data = open('/Users/kewang/CS294HW3/homework/hw3/results/64/PongKWANG7000001538896956.847053_data.pkl','rb')
Data = pkl.load(Data)
#Data_d = pkl.load(Data_d)
#%%
n = 750
time_step = np.array(Data['t_log'])-1000
mean_reward = Data['mean_episode_reward_log']
best_reward = Data['best_mean_episode_reward_log']
time_step = time_step[:n]
mean_reward = mean_reward[:n]
best_reward = best_reward[:n]

#
#time_step_d = np.array(Data_d['t_log'])-50000
#mean_reward_d = Data_d['mean_episode_reward_log']
#best_reward_d = Data_d['best_mean_episode_reward_log']
#time_step_d = time_step_d[:n]
#mean_reward_d = mean_reward_d[:n]
#best_reward_d = best_reward_d[:n]
#%%
#plt.plot(time_step,mean_reward,color = 'r', linestyle='-',linewidth = 2)

#plt.plot(time_step,best_reward,color = 'b', linestyle='-',linewidth = 2)
plt.plot(time_step,mean_reward,color = 'c', linestyle='-',linewidth = 2)
plt.plot(time_step,best_reward,color = 'k', linestyle='-',linewidth = 2)
plt.xticks([0,250000,500000,750000],fontsize=12)
plt.yticks(fontsize=12)
plt.ylim([-300,200])
plt.xlabel("Iteration times",fontsize = 18,labelpad = 0)
plt.ylabel("Policy reward",fontsize = 18,labelpad = 0)
plt.title("Deep Q Learning Performance on \n Lunar Lander (network size: 64)",fontsize = 18)
plt.legend(["best mean episode reward of DQN","best mean episode reward of DDQN"],fontsize = 12,loc ="center right")
plt.show()
