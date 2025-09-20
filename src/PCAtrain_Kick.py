# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 09:31:17 2024

@author: MA11201
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\a_PhD Research\SoftSensor\DrillData\Kick_Detection_Data2.csv")

data = data.rename(columns={'DPPressure': 'DPPress', 'CPressure':'CPress', 
                          'ROPenetration':'RoPen', 'WOBit':'WoBit', 
                          'WBoPressure':'WBoPress'})

cols = data.columns

for i in cols:
    fig, ax =plt.subplots(figsize=(6,4))
    ax.plot(data[i].iloc[10:])
    ax.set_title(i)
 
#CSDepth, BSize, FPress, CPress, MPS1, MPS2, MPS3, AMTD, STP, 
    

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats
import sklearn.decomposition
import sklearn.preprocessing
import lightgbm as lgb
from statsmodels.tsa.tsatools import lagmat
import sys
sys.path.append(r'C:\a_PhD Research\SoftSensor\Codes')
from PCAModel import *
import math

    
    
data1 = data.drop(['CSDepth', 'BSize', 'FPress', 'CPress', 'MPS1', 'MPS2', 'MPS3', 'AMTD', 'STP'], axis=1)
df_train = data1.iloc[10:1495, :]
#df_valid = data.iloc[1200:1495,:]    
df_test  = data1.iloc[1496:, :]



df_valid.isnull().values.sum()

model = ModelPCA()

spe_train = model.train(df_train, plot = True)

#spe_validation = model.test(df_valid, plot = True)

spe_test = model.test(df_test)

detection_limits = np.percentile(spe_train, 99.99)

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(spe_test)
ax.axhline(detection_limits, ls='--', color='red')
ax.grid()



fig, ax = plt.subplots(figsize=(6,4))
ax.plot(spe_train)
ax.axhline(detection_limits, ls='--', color='red')
ax.grid()


