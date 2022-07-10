#print('teste123 github Andre Monica')
#print('teste 456 github Ana Maria')
#print('Será que agora funciona?')
#print('Ola ANA :D teste a ver se consegues ver isto')
#https://www.kaggle.com/code/adinishad/95-accuracy-chest-x-ray-images-pneumonia
#https://www.kaggle.com/code/adinishad/95-accuracy-chest-x-ray-images-pneumonia
#print('Ola ANA :D teste a ver se consegues ver isto')

'''Temos de fazer import destas extensões: numpy (linear algebra), pandas (data processing) e cv2
no terminal coloquei->   pip install pandas       ->está a dar erro ao verificar o pip mas instalou as coisas
'''

#
# Actual code, MOVE WITH CAUTION!
#


#in [1]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np                  # linear algebra
import pandas as pd                 # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2                          # import cv2        >>> https://pypi.org/project/opencv-python/  || pip install opencv-python >> into CMD

#in [2]
import os
print(os.listdir("./img"))          # Folder with images must be in the same Domain as ou Main.py

#in [3]
DIR = os.listdir('./img/chest_xray') 
print(DIR)

#in [4]
#define the folder's content 
train_folder = './img/chest_xray/train'
test_folder = './img/chest_xray/test'
val_folder = './img/chest_xray/val'

#in [5]
import matplotlib.pyplot as plt     # matplotlib, self explanatory
import seaborn as sns               # seaborn, statiscical data visualization >>>  https://seaborn.pydata.org/ || pip install seaborn >> into CMD
from PIL import Image
import random

#in [6]
labels = ["NORMAL", "PNEUMONIA"]    # each folder has two sub folder name "PNEUMONIA", "NORMAL"
IMG_SIZE = 50                       # resize image

def get_data_train(data_dir): 
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([new_array, class_num], dtype=object)                       # !!!!! EVEN with Manually added dtype=object to resolve ERROR VisibleDeprecationWarning: 
                                                                                        # the behaviour seems to be off by a mile, list.append() takes no keyword arguments 
                                                                                        # *shown on terminal for each image scanned, aka WRITES exception for each image
            except Exception as e:
                print(e)
    return np.array(data)

#in [7] 
# get Data to train
train = get_data_train(train_folder)
test = get_data_train(test_folder)
val = get_data_train(val_folder)

#in [8]
l = []
for i in train:
    if(i[1] == 0):
       l.append("Normal")
    else:
       l.append("Pneumonia")
        
sns.countplot(l)

#in [9]
X_train = []
y_train = []

X_val = []
y_val = []

X_test = []
y_test = []

for feature, label in train:
    X_train.append(feature)
    y_train.append(label)

for feature, label in test:
    X_test.append(feature)
    y_test.append(label)

for feature, label in val:
    X_val.append(feature)
    y_val.append(label)