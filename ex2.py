import sys
import numpy as np
from numpy import random
import mnist
import matplotlib.pyplot as plt

random.seed(71)

units = 32
batch_size = 100
w1, b1 = random.normal(loc=0, scale=np.sqrt(1/784), size=784*units).reshape(units, 784), random.normal(loc=0, scale=np.sqrt(1/784), size=units)
w2, b2 = random.normal(loc=0, scale=np.sqrt(1/units), size=units*10).reshape(10, units), random.normal(loc=0, scale=np.sqrt(1/units), size=10)

train_x = mnist.download_and_parse_mnist_file('train-images-idx3-ubyte.gz')
train_y = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")

def preprocessing():
  idx = random.randint(0, len(train_y), batch_size)
  tr_x = train_x[idx]
  l = [[1 if i == label else 0 for i in range(10)] for label in train_y[idx]]
  tr_y = np.zeros((len(l), len(l[0])))
  tr_y[:] = l
  return tr_x, tr_y # tr_x is (bs, 28, 28) tr_y is one-hot-encoding (bs, 10)

def input_layer(im):
  return im.reshape((batch_size, 784)).T

def dense_layer(input_vec, weight, bias): # ミニバッチに対応
  return weight@input_vec+bias.reshape(-1, 1)

def dense_layer1(input_vec):
  return dense_layer(input_vec, w1, b1)

def sigmoid(input_vec):
  return 1/(1+np.exp(-input_vec))

def dense_layer2(input_vec):
  return dense_layer(input_vec, w2, b2)

def softmax(input_vec): # ミニバッチに対応
  return np.exp(input_vec-np.max(input_vec, axis=0)) / (np.sum(np.exp(input_vec-np.max(input_vec, axis=0)), axis=0))

def postprocessing(input_vec): # ミニバッチに対応
  return np.argmax(input_vec, axis=0)

def cross_entropy(true_vec ,pred_vec):
  return np.sum(np.sum(-1 * true_vec * np.log(pred_vec), axis=0)) / batch_size

def model():
  tr_x, tr_y = preprocessing()
  return cross_entropy(tr_y.T, softmax(dense_layer2(sigmoid(dense_layer1(input_layer(tr_x))))))

if __name__ == '__main__': 
  stdout = model()
  print(stdout)
