# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 12:43:41 2024

@author: MA11201
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats
import sklearn.decomposition
import sklearn.preprocessing
import lightgbm as lgb

class ModelPCA():
    
    def spe (self, X, X_pred): 
        return np.sum((X-X_pred)**2, axis=1)
   
    def train(self, df_train, plot = False):
        
        self.mu_train = df_train.mean(axis=0)
        self.std_train = df_train.std(axis=0)

        self.m = sklearn.decomposition.PCA(n_components = 0.9)

        X_train = sklearn.preprocessing.scale(df_train)
        X_train_pred = self.m.inverse_transform(self.m.fit_transform(X_train))
        
        if plot:
            fig, ax = plt.subplots()
            xaxis = np.arange(len(self.m.explained_variance_ratio_))
            ax.bar(xaxis, self.m.explained_variance_ratio_)
            ax.plot(xaxis, np.cumsum(self.m.explained_variance_ratio_));
            ax.set_title('PCA - Explained variance');
        
        return self.spe(X_train, X_train_pred)
            
    def test(self, df_test, plot = False):
        
        X_test = np.array((df_test-self.mu_train)/self.std_train)
        X_test_pred = self.m.inverse_transform(self.m.transform(X_test))

        return self.spe(X_test, X_test_pred)
    
    

def apply_lag (df, lag = 1):
       
    from statsmodels.tsa.tsatools import lagmat
    array_lagged = lagmat(df, maxlag=lag,
                          trim="forward", original='in')[lag:,:]  
    new_columns = []
    for l in range(lag):
        new_columns.append(df.columns+'_lag'+str(l+1))
    columns_lagged = df.columns.append(new_columns)
    index_lagged = df.index[lag:]
    df_lagged = pd.DataFrame(array_lagged, index=index_lagged,
                             columns=columns_lagged)
       
    return df_lagged


def filter_noise_ma (df, WS = 100,reduction = False):

    import copy
    
    new_df = copy.deepcopy(df)

    for column in df:
        new_df[column] = new_df[column].rolling(WS).mean()

    if reduction:
        return new_df.drop(df.index[:WS])[::WS]
    else:
        return new_df.drop(df.index[:WS])



