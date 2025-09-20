# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 10:39:34 2024

@author: MA11201
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


df = pd.read_csv(r"C:\a_PhD Research\SoftSensor\DrillData\Kick_Detection_Data2.csv")

df.info()

fig, ax = plt.subplots(figsize=(20, 19))
sns.heatmap(abs(df.corr(method='pearson')), cmap='summer', annot=True, fmt='.2f', ax=ax)
plt.savefig(r'C:\a_PhD Research\SoftSensor\Figures\Correlation_all_vars_kick.pdf')
plt.show()


#Plotting Mutual Information for all the variables
MI = [[0 for x in range(29)] for y in range(29)]
cols1 = df.columns
cols2 = df.columns
for i in range(len(cols1)):
    for j in range(len(cols2)):
        MI[i][j] = mutual_info_score(df[cols1[i]], df[cols2[j]])
        
MI1 = np.array(MI)
fig, ax = plt.subplots(figsize=(20, 19))
sns.heatmap(MI1, cmap='summer', annot=True, fmt='.2f', ax=ax)
plt.savefig(r'C:\a_PhD Research\SoftSensor\Figures\MI_kick.pdf')
plt.show()


df1 = MinMaxScaler().fit_transform(df)
df2 = pd.DataFrame(df1)
df2.columns = df.columns


df2.info()




df3 = df2[['FRate', 'SMSpeed', 'FIn', 'FOut', 'MRFlow', 
           'ActiveGL', 'ATVolume', 'ROPenetration', 'WOBit', 
           'HLoad', 'WBoPressure', 'BTBR', 'ATMPV', 'ATMYP',
           'WellDepth', 'FPress']]






df3.plot()


for i in df3.columns:
    df3[[i, 'ActiveGL']].plot()


















