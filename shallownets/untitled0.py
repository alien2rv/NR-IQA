# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 12:50:49 2019

@author: rravela
"""
res = gblur_model("D:\\thesis\\coding\\shallownets\\gblur.h5")
X_test = jpeg2000_lcn[141:]
label_test = label[141:]
X_test,Y_test = image_split(X_test,32,label_test)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
pred = res.predict(X_test)
Y_pred = []
s=0
count = 0
for i in range(1,len(Y_test),1):
    if (Y_test[i-1,2] == Y_test[i,2]):
        s = s+pred[i,0]
        count = count+1
    else:
        mean = s/count
        Y_pred.append(mean)
        s=0
        count=0
Y_pred.append(s/count)

from scipy import stats

PLCC = stats.pearsonr(label_test[:,0], Y_pred)
SROCC = stats.spearmanr(label_test[:,0], Y_pred)

np.save("D:\\thesis\\coding\\shallownets\\nps\\blur_predicted.npy", Y_pred, allow_pickle=True, fix_imports=True)
np.save("D:\\thesis\\coding\\shallownets\\nps\\blur_test.npy", label_test, allow_pickle=True, fix_imports=True)

print(PLCC)
print(SROCC)

import matplotlib.pyplot as plt
plt.scatter(label_test[:,0], Y_pred)