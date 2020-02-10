# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 23:04:03 2018

@author: pqian
"""
%load_ext autoreload
%autoreload 2
import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
#import sklearn


v_python_script_path = r'H:\BU2016\Folder\Python Script\Into System Path'
master_folder = r'H:\BU2016\Folder\Project\027_ETG Modeling'
os.chdir(master_folder)
import conf_var
import conf_var_dc
dataFile_path = 'ETG Joint Degeneration Data Set.txt'
#-----------------------
#get dummy vairables for
#    1. gender
#    2. age band
#    3. in network indicator
#    4. episode start weekday
#    5. race code
#-----------------------
col_dummy = ['GENDER', 'AGE_BAND', 'IN_NETWORK_IND', 'EPSD_STT_WK', 'RACE_CD', 'MEMBER_CNTY', 'PROV_CNTY']
#Imports from Python Scripts folder
sys.path.append(v_python_script_path)
import machine_learning_fxn as mlf
df_etg = mlf.read_data(dataFile_path)
df_etg = mlf.add_dummies(df_etg, col_dummy)
df_etg['TOT_ALLOWED'] = np.log(df_etg['TOT_ALLOWED']) # log transform Y variable
###
#Look at descriptive stats, the decide how to manipulate data
###
etg_desc = df_etg.describe()
# create a new age brands (aggregating some with low means into one)
df_etg['AGE_BAND_[40-64]'] = df_etg[['AGE_BAND_[40-44]', 'AGE_BAND_[45-49]', 'AGE_BAND_[55-59]', 'AGE_BAND_[60-64]']].sum(axis=1)
df_etg['AGE_BAND_[80+]'] = df_etg[['AGE_BAND_[80-84]', 'AGE_BAND_[85+]']].sum(axis=1)
df_etg['EPSD_STT_WK_Weekend'] = df_etg[['EPSD_STT_WK_Saturday', 'EPSD_STT_WK_Sunday']].sum(axis=1)
df_etg['MEMBER_CNTY_REST_OF_MICHIGAN'] = df_etg[['MEMBER_CNTY_LIVINGSTON', 'MEMBER_CNTY_MONROE', 'MEMBER_CNTY_REST OF MICHIGAN', 'MEMBER_CNTY_ST. CLAIR','MEMBER_CNTY_WASHTENAW']].sum(axis=1)
df_etg['PROV_CNTY_REST_OF_MICHIGAN'] = df_etg[['PROV_CNTY_LIVINGSTON', 'PROV_CNTY_MONROE', 'PROV_CNTY_REST OF MICHIGAN', 'PROV_CNTY_ST. CLAIR','PROV_CNTY_WASHTENAW']].sum(axis=1)
# Split data set into two, by with or without IP unils
df_etg_wIP = df_etg[df_etg['IP_UTIL']==1]
df_etg_woIP = df_etg[df_etg['IP_UTIL']==0]
###
#Look at descriptive stats, Close
###
# get X and Y variables for modeling
Y = mlf.get_model_variable(df_etg, 'TOT_ALLOWED')
X = mlf.get_model_variable(df_etg, conf_var.var_list)
Y_wIP = mlf.get_model_variable(df_etg_wIP, 'TOT_ALLOWED')
X_wIP = mlf.get_model_variable(df_etg_wIP, conf_var.var_list)
Y_woIP = mlf.get_model_variable(df_etg_woIP, 'TOT_ALLOWED')
X_woIP = mlf.get_model_variable(df_etg_woIP, conf_var.var_list)
mlf.y_x_plot_scatter(Y, X[['IP_UTIL', 'ER_UTIL']])
mlf.chk_del_field_same_val(X)
#multicollinearity test - Eigenvalues, Eigenvector
mulcol_eng, v = mlf.multicol_eigen(X_wIP)
#multicollinearity test - VIF 
mulcol_vif = mlf.multicol_vif(X_wIP)
#run statistical model
mlf.statistical_regression(X_wIP, Y_wIP, 'OLS')

mlf.feature_selection(X, Y, None, 'Regression')
"""
### It looks that the linear model fits the data very well (explaining the data in a good way)
### Then let's see how good the model could predict 
### that MI algorithms could generate with the data set
"""
#run machine learning algorithm - decision tree
feature_name = X.columns
df_pred_output = mlf.ml_regressor(X_wIP, Y_wIP, 'Decision Tree', feature_name)
df_pred_output


import matplotlib.pyplot as plt
plt.scatter(Y, X['IP_UTIL'])