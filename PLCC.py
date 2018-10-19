# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 09:28:40 2018

@author: ravi
"""

import numpy as np
y_train = np.load('y_train.npy')
y_test = np.load('D:\\cnn\\results\\val_split_0.4\\32epcs.npy')

temp = []

for i in range(0,y_train.shape[0]):
    if (y_train[i-1] != y_train[i]):
        temp.append(i)   
y_valid = []
y_orig = []
for i in range (1,145):
    block = y_test[temp[i-1]:temp[i]]
    block = np.array(np.mean(block))
    y_valid.append(block)
    
    block = y_train[temp[i-1]:temp[i]]
    block = np.array(np.mean(block))
    y_orig.append(block)   
    
    
       

y_orig = np.array(y_orig)
y_valid = np.array(y_valid)


from scipy import stats

PLCC = stats.pearsonr(y_orig, y_valid)

#results_vs1 = []
results_vs1.append(PLCC[0])


#result_vs3 = np.array(results_vs1)
#np.save('result_vs3.npy',result_vs3)