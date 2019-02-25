# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 14:11:53 2019

@author: RRavela
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

blur = np.load("D:\\SPIE2019\\coding\\results\\live\\blur_predicted.npy", allow_pickle=True, fix_imports=True)
awgn = np.load("D:\\SPIE2019\\coding\\results\\live\\awgn_predicted.npy", allow_pickle=True, fix_imports=True)
jpeg = np.load("D:\\SPIE2019\\coding\\results\\live\\jpeg_predicted.npy", allow_pickle=True, fix_imports=True)
jp2k = np.load("D:\\SPIE2019\\coding\\results\\live\\jp2k_predicted.npy", allow_pickle=True, fix_imports=True)

jp2k_original = np.load("D:\\SPIE2019\\coding\\results\\live\\jp2k_test.npy", allow_pickle=True, fix_imports=True)
jpeg_original = np.load("D:\\SPIE2019\\coding\\results\\live\\jpeg_test.npy", allow_pickle=True, fix_imports=True)
awgn_original = np.load("D:\\SPIE2019\\coding\\results\\live\\awgn_test.npy", allow_pickle=True, fix_imports=True)
blur_original = np.load("D:\\SPIE2019\\coding\\results\\live\\blur_test.npy", allow_pickle=True, fix_imports=True)

g1 = (jpeg,jpeg_original[:,0])
g2 = (jp2k,jp2k_original[:,0])
g3 = (blur,blur_original[:,0])
g4 = (awgn,awgn_original[:,0])

data = (g1, g2, g3, g4)

colors = ("red", "green", "blue")
groups = ("jpeg", "jp2k", "gblur")

# Create plot
fig = plt.figure()
ax1 = fig.add_subplot(111)
 
ax1.scatter(jpeg,jpeg_original, s=10, c='b', marker="o", label='jpeg')
ax1.scatter(jp2k,jp2k_original, s=10, c='r', marker="o", label='jpeg2000')
ax1.scatter(blur,blur_original, s=10, c='g', marker="o", label='blur')
ax1.scatter(awgn,awgn_original, s=10, c='y', marker="o", label='awgn')
plt.legend(loc='lower right');
plt.show()

PLCC = stats.pearsonr(blur, blur_original)
SROCC = stats.spearmanr(jpeg, jpeg_original)