# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 13:16:03 2024

@author: MA11201
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mutual_info_score as mis

df1 = pd.read_csv(r"C:\a_PhD Research\SoftSensor\DrillData\FormationChangeData.csv")

df1 = df1.drop(['CSDepth'], axis=1)



df1 = df1.rename(columns={'DPPressure': 'DPPress', 'CPressure':'CPress', 
                          'ROPenetration':'RoPen', 'WOBit':'WoBit', 
                          'WBoPressure':'WBoPress'})

df_corr = df1[['FPress', 'WellDepth', 'BTBR', 'WBoPress', 'HLoad', 'WoBit', 'RoPen', 'DPPress']]
corr = abs(df_corr.corr(method='pearson'))
matrix = np.triu(corr)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(corr, ax=ax, cmap='crest', annot=True, fmt='.2f', cbar=False, mask=matrix)
plt.savefig(r'C:\a_PhD Research\SoftSensor\Figures\Scientific data Figs\FPR.svg', bbox_inches="tight")


cols = df1.columns
row_list=[]
for i in range(len(cols)):
    row_temp = []
    col_a = df1[cols[i]]
    for j in range(len(cols)):
        col_b = df1[cols[j]]
        m_info = mis(col_a, col_b)
        row_temp.append(m_info)
    row_list.append(row_temp)
m_info_mat = np.vstack(row_list)


mi_mat = pd.DataFrame(m_info_mat)
mi_mat.columns = cols
mi_mat.index =cols




mi_mat1 = mi_mat[['FPress', 'WellDepth', 'BTBR', 'WBoPress', 'HLoad', 'WoBit', 'RoPen', 'MRFlow', 'FOut', 'FIn', 'DPPress', 'BDepth', 'SMSpeed', 'FRate', 'CircFlow']]

cols_new = ['FPress', 'WellDepth', 'BTBR', 'WBoPress', 'HLoad', 'WoBit', 'RoPen', 'MRFlow', 'FOut', 'FIn', 'DPPress', 'BDepth', 'SMSpeed', 'FRate', 'CircFlow']

new_list= []
for c in cols_new:
    new_list.append(mi_mat1.loc[c,:])

new_list1 = np.vstack(new_list)

mi_mat2 = pd.DataFrame(new_list1)
mi_mat2.columns = cols_new
mi_mat2.index= cols_new

matrix = np.triu(mi_mat2)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(mi_mat2, ax=ax, cmap='crest', annot=True, fmt='.2f', cbar=False, mask=matrix)
plt.savefig(r'C:\a_PhD Research\SoftSensor\Figures\Scientific data Figs\FPR_MI.svg', bbox_inches="tight")





