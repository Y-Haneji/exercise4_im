import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.backend import set_session, tensorflow_backend

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0'))
set_session(tf.Session(config=config))

img_rows, img_cols = 28, 28
num_classes = 10

(X, Y), (Xtest, Ytest) = keras.datasets.mnist.load_data()
X = X.reshape(X.shape[0],img_rows,img_cols,1)
Xtest = Xtest.reshape(X.shape[0],img_rows,img_cols,1)

# 正規化
X = X.astype('float32')/255.0
Xtest = Xtest.astype('float32')/255.0

input_shape = (img_rows,img_cols,1)

Y = keras.utils.to_categorical(Y, num_classes)
Ytest1 = keras.utils.to_categorical(Ytest, num_classes)

model = models.Sequential()
model.add(layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape, padding='same'))
model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation=
'softmax'))

print(model.summary())

model.compile(
  loss=keras.losses.categorical_crossentropy,
  optimizer=keras.optimizer.Adam(),
  metrics=['acc'])

epochs=50
batch_size=32
result = model.fit(X,Y,batch_size=batch_size, epochs=epochs, validation_data=(Xtest, Ytest1))
