# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 20:43:03 2024

@author: MA11201
"""

import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r"C:\a_PhD Research\SoftSensor\DrillData\FormationChangeData.csv")
df.info(verbose=True)


#Plotting correlations for all the variables
#Pearson correlation is used because Formation pressure is linearly related with other variables (see pairplot)
fig, ax = plt.subplots(figsize=(20, 19))
sns.heatmap(abs(df.corr(method='pearson')), cmap='summer', annot=True, fmt='.2f', ax=ax)
#plt.savefig(r'C:\a_PhD Research\SoftSensor\Figures\Correlation_all_vars.pdf')
#plt.show()

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
#plt.savefig(r'C:\a_PhD Research\SoftSensor\Figures\MI.pdf')
#plt.show()
    
#Plotting Cosine Similarity
CS = [[0 for x in range(29)] for y in range(29)]
for i in range(len(cols1)):
    vector_a = np.array(df[cols1[i]])
    vector_a1 = vector_a.reshape(-1, 1)
    for j in range(len(cols2)):
        vector_b = np.array(df[cols2[j]])
        vector_b1 = vector_b.reshape(-1, 1)
        CS[i][j] = cosine_similarity(vector_a1, vector_b1)


CS1 = np.array(CS)
fig, ax = plt.subplots(figsize=(20, 19))
sns.heatmap(CS1, cmap='summer', annot=True, fmt='.2f', ax=ax)
#plt.savefig(r'C:\a_PhD Research\SoftSensor\Figures\CS.pdf')
#plt.show()


df['FPress'] = df['FPress'].apply(lambda x: math.ceil(x))
plt.plot(df['FPress'])

#Mutual Information from different layer of formation
FormPress = df['FPress'].unique()

data_fpress = []
for i in range(len(FormPress)):
    dummy_data =  df[df['FPress']==FormPress[i]]
    data_fpress.append(dummy_data)
    



MI_global = []
for k in range(len(data_fpress)):
    MI_local = [[0 for x in range(29)] for y in range(29)]
    cols1 = df.columns
    cols2 = df.columns
    for i in range(len(cols1)):
        for j in range(len(cols2)):
            MI_local[i][j] = mutual_info_score(data_fpress[k][cols1[i]], data_fpress[k][cols2[j]])
            
    MI_global.append(MI_local)




CS_Global = []
for k in range(len(data_fpress)):
    CS_local = [[0 for x in range(29)] for y in range(29)]
    for i in range(len(cols1)):
        vector_a = np.array(data_fpress[k][cols1[i]])
        vector_a1 = vector_a.reshape(-1, 1)
        for j in range(len(cols2)):
            vector_b = np.array(data_fpress[k][cols2[j]])
            vector_b1 = vector_b.reshape(-1, 1)
            CS_local[i][j] = cosine_similarity(vector_a1, vector_b1)
            print(CS_local[i][j])
    CS_Global.append(CS_local)







fig, ax = plt.subplots(figsize=(20, 19))
sns.heatmap(CS_Global[0], cmap='summer', annot=True, fmt='.2f', ax=ax)



fig, ax = plt.subplots(figsize=(6,5))
ax.scatter(np.array(data_fpress[2]['FPress']), np.array(data_fpress[2]['DPPressure']))





    
