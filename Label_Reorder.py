import numpy as np
import os
import pandas as pd

folder = 'D:\gblur'
dataset = pd.read_csv('blr.csv', header=None)
y=dataset.values.tolist()


imgnames=[]
for filename in os.listdir(folder):
    imgnames.append(filename)

for i in range(len(y)):
    y[i].append(imgnames[i])
for i in y:
    print(i[1])
    
k = pd.read_csv('D:\\Databases\\id2013\\mos_with_names.txt', header=None, delimiter = ' ')
y = k.iloc[:,:].values


folder = 'D:\\Databases\\id2013\\distorted_images'
i=0
for filename in os.listdir(folder):
            #img = cv2.imread((os.path.join(folder,filename)), cv2.IMREAD_COLOR).astype(np.float32)
            if(filename==y[i,1]):
                print('yes')
            i = i+1


