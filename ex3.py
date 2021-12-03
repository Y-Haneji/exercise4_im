import sys
from tqdm import tqdm
import numpy as np
from numpy import random
import mnist
import matplotlib.pyplot as plt

random.seed(71)

class Model():
  def __init__(self) -> None:
    # モデルのアーキテクチャを作成
    self.units = 32
    self.batch_size = 100
    self.w1, self.b1 = random.normal(loc=0, scale=np.sqrt(1/784), size=784*self.units).reshape(self.units, 784), random.normal(loc=0, scale=np.sqrt(1/784), size=self.units)
    self.w2, self.b2 = random.normal(loc=0, scale=np.sqrt(1/self.units), size=self.units*10).reshape(10, self.units), random.normal(loc=0, scale=np.sqrt(1/self.units), size=10)

  def preprocessing(self, train_x, train_y):
    idx = random.randint(0, len(train_y), self.batch_size)
    tr_x = train_x[idx]
    l = [[1 if i == label else 0 for i in range(10)] for label in train_y[idx]]
    tr_y = np.zeros((len(l), len(l[0])))
    tr_y[:] = l
    return tr_x, tr_y # tr_x is (bs, 28, 28) tr_y is one-hot-encoding (bs, 10)

  def input_layer(self, im):
    return im.reshape((self.batch_size, 784)).T

  def dense_layer(self, input_vec, weight, bias): # ミニバッチに対応
    return weight@input_vec+bias.reshape(-1, 1)

  def dense_layer1(self, input_vec):
    return self.dense_layer(input_vec, self.w1, self.b1)

  def sigmoid(self, input_vec):
    return 1/(1+np.exp(-input_vec))

  def dense_layer2(self, input_vec):
    return self.dense_layer(input_vec, self.w2, self.b2)

  def softmax(self, input_vec): # ミニバッチに対応
    return np.exp(input_vec-np.max(input_vec, axis=0)) / (np.sum(np.exp(input_vec-np.max(input_vec, axis=0)), axis=0))

  def postprocessing(self, input_vec): # ミニバッチに対応
    return np.argmax(input_vec, axis=0)

  def cross_entropy(self, true_vec ,pred_vec):
    return np.sum(np.sum(-1 * true_vec * np.log(pred_vec), axis=0)) / self.batch_size

  def train_batch(self, tr_x, tr_y, lr):
    x = self.input_layer(tr_x)
    t = self.dense_layer1(x)
    y = self.sigmoid(t)
    a = self.dense_layer2(y)
    y2 = self.softmax(a)
    e_n = self.cross_entropy(tr_y.T, y2)

    # 以下微分の形を行*列で示す
    diff_e_n_with_a = (y2 - tr_y.T)/self.batch_size # クラス数*bs
    diff_e_n_with_y = self.w2.T @ diff_e_n_with_a # units*bs
    diff_e_n_with_w2 = diff_e_n_with_a @ y.T # クラス数*units
    diff_e_n_with_b2 = np.sum(diff_e_n_with_a, axis=1) # クラス数次元ベクトル
    diff_e_n_with_t = diff_e_n_with_y*(1-y)*y # units*bs
    diff_e_n_with_x = self.w1.T @ diff_e_n_with_t # 784*bs
    diff_e_n_with_w1 = diff_e_n_with_t @ x.T # units*784
    diff_e_n_with_b1 = np.sum(diff_e_n_with_t, axis=1) # units数次元ベクトル

    # 重みの更新
    self.w1 = self.w1 - lr*diff_e_n_with_w1
    self.b1 = self.b1 - lr*diff_e_n_with_b1
    self.w2 = self.w2 - lr*diff_e_n_with_w2
    self.b2 = self.b2 - lr*diff_e_n_with_b2

    return e_n
  
  def train(self, train_x, train_y, epochs: int = 10, lr: float = 0.01):
    '''
    train_x: 学習データの特徴量
    train_y: 学習データのラベル
    epochs: エポック数
    lr: 学習率
    '''
    for i in tqdm(range(epochs)):
      for j in range(60000//self.batch_size):
        tr_x, tr_y = self.preprocessing(train_x, train_y)
        res = self.train_batch(tr_x, tr_y, lr)
      print(f'Epoch {i+1} end! Cross entroppy is {res}.')


if __name__ == '__main__': 
  train_x = mnist.download_and_parse_mnist_file('train-images-idx3-ubyte.gz')
  train_y = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")
  model = Model()
  model.train(train_x, train_y)
