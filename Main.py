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
                data.append([new_array, class_num])                                     # !!!!! EVEN with Manually added dtype=object to resolve ERROR VisibleDeprecationWarning: 
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
        
sns.countplot(l)                        # shows graphplot

#out [8]    <matplotlib.axes._subplots.AxesSubplot at 0x7fa43ba0f750>   !!! missing

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
    
#in [10]
X_train = np.array(X_train) / 255
X_val = np.array(X_val) / 255
X_test = np.array(X_test) / 255

#in [11]
X_test.shape

#out [11] (624, 50, 50)

#in [12]
X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = np.array(y_train)

X_val = X_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_val = np.array(y_val)

X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = np.array(y_test)

#in [13]
# https://www.tensorflow.org/install
# Requires the latest pip >>> pip install --upgrade pip
# Current stable release for CPU and GPU >>>  pip install tensorflow
# Or try the preview build (unstable) >>> pip install tf-nightly
# all into CMD

import tensorflow as tf
from tensorflow.keras.layers import Flatten, Conv2D, Activation, Dense, Dropout, MaxPooling2D   #issues with this RED!!!
from tensorflow.keras.models import Sequential                                                  #issues with this RED!!!

# tentei para resolver os de cima a vermelho 
# >>pip3 install ––upgrade setuptools
# >>python -m pip install --upgrade pip
# nao deu em nada
# tensorflow 2.9.1
# Keras 2.9.0



#in [14]
model = Sequential()

model.add(Conv2D(32, (3, 3), padding="same", input_shape=X_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation="relu"))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=25, validation_data=(X_val, y_val), shuffle=True)
scores = model.evaluate(X_test, y_test)

model.save("cnn.model")

#in [15]
# scores
print("Test loss {}".format(scores[0]))
print("Test accuracy {}".format(scores[1]))

# expected results
#Test loss 1.0829148292541504
#Test accuracy 0.8108974099159241


#in [16]
# visualization

import matplotlib.pyplot as plt
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))

plt.plot(epochs, accuracy, "b", label="trainning accuracy")
plt.plot(epochs, val_accuracy, "r", label="validation accuracy")
plt.legend()
plt.show()

plt.plot(epochs, loss, "b", label="trainning loss")
plt.plot(epochs, val_loss, "r", label="validation loss")
plt.legend()
plt.show()

#in [17]
# predict classes

prediction = model.predict_classes(X_test)
prediction = prediction.reshape(1, -1)[0]
prediction[:15]

#out [17] array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=int32)

#in [18]
# correct and incorrect
# you can check tensorflow website

correct = np.nonzero(prediction == y_test)[0]
incorrect = np.nonzero(prediction != y_test)[0]

#in [19] Visualize some correct
j = 0
for i in correct[:6]:
    plt.subplot(3,2,j+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_test[i].reshape(50,50), cmap="gray", interpolation='none')
    plt.title("Predicted Class {},Actual Class {}".format(prediction[i], y_test[i]))
    plt.xlabel(labels[prediction[i]])
    plt.tight_layout()
    j += 1

#in [20] Some inccorect visualization
j = 0
for i in incorrect[:6]:
    plt.subplot(3,2,j+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_test[i].reshape(50,50), cmap="gray", interpolation='none')
    plt.title("Predicted Class {},Actual Class {}".format(prediction[i], y_test[i]))
    plt.xlabel(labels[prediction[i]])
    plt.tight_layout()
    j += 1
    
#in [21]
# load model and predict some some external photo

labels = ["NORMAL", "PNEUMONIA"]
def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("cnn.model") # load model

#in [22] !!!!!!!!!!!!!!!!!!!!!! look into this
# extra pneumonia photo from google
prediction = model.predict([prepare("./img/protest/left-lower-lobe-pneumonia.jpg")])
print(labels[int(prediction[0])])

#in [23] !!!!!!!!!!!!!!!!!!!!!! look into this
# extra normal x-ray photo from google
prediction = model.predict([prepare("./img/protest/normal.jpeg")])
print(labels[int(prediction[0])])

#in [24] !!!!!!!!!!!!!!!!!!!!!! look into this
prediction = model.predict([prepare("./img/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg")])
print(labels[int(prediction[0])])

#in [25] !!!!!!!!!!!!!!!!!!!!!! look into this
prediction = model.predict([prepare("./img/chest_xray/test/PNEUMONIA/person101_bacteria_486.jpeg")])
print(labels[int(prediction[0])])