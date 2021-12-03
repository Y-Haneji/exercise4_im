import sys
import numpy as np
from numpy import random
import mnist
import matplotlib.pyplot as plt

random.seed(71)
units = 32
w1, b1 = random.normal(loc=0, scale=np.sqrt(1/784), size=784*units).reshape(units, 784), random.normal(loc=0, scale=np.sqrt(1/784), size=units)
w2, b2 = random.normal(loc=0, scale=np.sqrt(1/units), size=units*10).reshape(10, units), random.normal(loc=0, scale=np.sqrt(1/units), size=10)

def preprocessing(input):
  return mnist.download_and_parse_mnist_file('train-images-idx3-ubyte.gz')[input]

def input_layer(im):
  return im.flatten().reshape(-1, 1)

def dense_layer(input_vec, weight, bias):
  return weight@input_vec+bias.reshape(-1, 1)

def dense_layer1(input_vec):
  return dense_layer(input_vec, w1, b1)

def sigmoid(input_vec):
  return 1/(1+np.exp(-input_vec))

def dense_layer2(input_vec):
  return dense_layer(input_vec, w2, b2)

def softmax(input_vec):
  return np.exp(input_vec-np.max(input_vec, axis=0)) / (np.sum(np.exp(input_vec-np.max(input_vec, axis=0))))

def postprocessing(input_vec):
  return np.argmax(input_vec)

def model(input):
  return postprocessing(softmax(dense_layer2(sigmoid(dense_layer1(input_layer(preprocessing(input)))))))

if __name__ == '__main__':  
  stdin = int(input())
  stdout = model(stdin)
  # stdout = dense_layer1(input_layer(preprocessing(stdin))).shape
  print(stdout)
