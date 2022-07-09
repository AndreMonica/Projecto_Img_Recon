#print('teste123 github Andre Monica')
#print('teste 456 github Ana Maria')
#print('Será que agora funciona?')
#print('Ola ANA :D teste a ver se consegues ver isto')
#https://www.kaggle.com/code/adinishad/95-accuracy-chest-x-ray-images-pneumonia
#https://www.kaggle.com/code/adinishad/95-accuracy-chest-x-ray-images-pneumonia


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