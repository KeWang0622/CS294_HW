#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 18:24:59 2018

@author: kewang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 18:02:42 2018

@author: kewang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 17:32:49 2018

@author: kewang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 14:23:47 2018

@author: kewang
"""

#%%Plot Curves for rewards under different tasks 
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import pandas as pd
import seaborn as sns

#%% Import the rewards data from csv files.
filenames = os.listdir('/Users/kewang/CS294HW/homework/hw1/Results_Dagger_Hopper')

num_echo = []
data_mean = []
data_std = []

num_echo_e = []
data_mean_e = []
data_std_e = []
for filename in filenames:
    if filename[0:7] == 'dagger_':
        print(filename)
        num_echos = int(filename[17:-11])
        print(num_echos)
        list_raw = list(pd.read_csv('/Users/kewang/CS294HW/homework/hw1/Results_Dagger_Hopper/'+filename))[0:-2]
        data_raw = []
        for i in range(len(list_raw)):
            data_raw.append(float(list_raw[i]))
        data_mean.append(np.mean(data_raw))
        data_std.append(np.std(data_raw))
        num_echo.append(num_echos)

    if filename[0:7] == 'daggere':
        print(filename)
        num_echos = int(filename[18:-11])
        print(num_echos)
        list_raw = list(pd.read_csv('/Users/kewang/CS294HW/homework/hw1/Results_Dagger_Hopper/'+filename))[0:-2]
        data_raw = []
        for i in range(len(list_raw)):
            data_raw.append(float(list_raw[i]))
        data_mean_e.append(np.mean(data_raw))
        data_std_e.append(np.std(data_raw))
        num_echo_e.append(num_echos)
#%% Sort
num_echo_sort = np.sort(num_echo)
data_mean_sort = []
data_std_sort = []

num_echo_sort_e = np.sort(num_echo_e)
data_mean_sort_e = []
data_std_sort_e = []
for i in num_echo_sort:
    n = num_echo.index(i)
    data_mean_sort.append(data_mean[n])
    data_std_sort.append(data_std[n])

for i in num_echo_sort_e:
    n = num_echo_e.index(i)
    data_mean_sort_e.append(data_mean_e[n])
    data_std_sort_e.append(data_std_e[n])

#%% Plot the curves
plt.plot(np.log10(num_echo_sort),data_mean_sort,color = 'r', linestyle='-',linewidth = 2, marker = '*',markersize = 10)
plt.errorbar(np.log10(num_echo_sort),data_mean_sort,data_std_sort,color = 'r',fmt = '.r',linewidth = 2)
plt.plot(np.log10(num_echo_sort_e),data_mean_sort_e,color = 'b', linestyle='-',linewidth = 2, marker = '.',markersize = 10)
plt.errorbar(np.log10(num_echo_sort_e),data_mean_sort_e,data_std_sort_e,color = 'b',fmt = '.b',linewidth = 2)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Log 10 (Number of Iterations)",fontsize = 18,labelpad = 0)
plt.ylabel("Policy's Reward",fontsize = 18,labelpad = 0)
plt.title("Dagger Performance over Epochs\n Number (Hopper Task)",fontsize = 18)
plt.legend(["Our reward","Expert's reward"],fontsize = 12,loc ="center left")
plt.show()


