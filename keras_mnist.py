import sys
from tqdm import tqdm
import numpy as np
from numpy import random
import mnist
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

train_x = mnist.download_and_parse_mnist_file('train-images-idx3-ubyte.gz')
train_y = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")
test_x = mnist.download_and_parse_mnist_file('t10k-images-idx3-ubyte.gz')
test_y = mnist.download_and_parse_mnist_file('t10k-labels-idx1-ubyte.gz')

from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.backend import sigmoid, softmax
from tensorflow.keras.utils import to_categorical

model = Sequential()
model.add(Dense(32, input_shape=(28*28,), activation=sigmoid))
model.add(Dense(10, activation=softmax))
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

train_x = train_x.reshape((train_x.shape[0], -1))
train_y = to_categorical(train_y, num_classes=10)
test_x = test_x.reshape((test_x.shape[0], -1))
model.fit(train_x, train_y, batch_size=100, epochs=10, verbose=1)

pred_y = model.predict(test_x)
pred_y = np.argmax(pred_y, axis=1)
print(accuracy_score(test_y, pred_y))
