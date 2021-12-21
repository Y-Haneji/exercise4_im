import sys
from tqdm import tqdm
import numpy as np
from numpy import random
import math
import mnist
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from logger import Logger

random.seed(71)

class Model():
  def __init__(self, mode = 'training', dropout: float = 0.0) -> None:
    # モデルのアーキテクチャを作成
    self.units = 32
    self.batch_size = 100
    self.filter_num = 32
    self.filter_w = 3
    self.filter_h = 3
    self.pad_w = math.floor(self.filter_w/2)
    self.pad_h = math.floor(self.filter_h/2)
    self.im_w = 28
    self.im_h = 28
    self.output_x = self.im_w-self.filter_w+2*self.pad_w+1
    self.output_y = self.im_h-self.filter_h+2*self.pad_h+1
    self.w1, self.b1 = random.normal(loc=0, scale=np.sqrt(1/784), size=784*self.units).reshape(self.units, 784), random.normal(loc=0, scale=np.sqrt(1/784), size=self.units)
    # self.w2, self.b2 = random.normal(loc=0, scale=np.sqrt(1/self.units), size=self.units*10).reshape(10, self.units), random.normal(loc=0, scale=np.sqrt(1/self.units), size=10)
    self.w2, self.b2 = random.normal(loc=0, scale=np.sqrt(1/self.filter_num*self.output_y*self.output_x), size=self.filter_num*self.output_y*self.output_x*10).reshape(10, self.filter_num*self.output_y*self.output_x), random.normal(loc=0, scale=np.sqrt(1/self.filter_num*self.output_y*self.output_x), size=10)
    self.filter, self.filter_bias = random.randn(self.filter_num, self.filter_h*self.filter_w), random.randn(self.filter_num)
    if mode not in ['train', 'inference']:
      raise ValueError('mode is train or inference.')
    self.mode = mode
    self.drop_prob = dropout

  def preprocessing(self, train_x, train_y):
    '''学習用セットからバッチを作成する'''
    idx = random.randint(0, len(train_y), self.batch_size)
    tr_x = train_x[idx]
    l = [[1 if i == label else 0 for i in range(10)] for label in train_y[idx]]
    tr_y = np.zeros((len(l), len(l[0])))
    tr_y[:] = l
    return tr_x, tr_y # tr_x is (bs, 28, 28) tr_y is one-hot-encoding (bs, 10)

  def input_layer(self, im):
    return im  # (bs, 28, 28)

  def dense_layer(self, input_vec, weight, bias): # ミニバッチに対応
    return weight@input_vec+bias.reshape(-1, 1)

  def dropout(self, input_vec):
    if self.mode == 'train':
      dropped = np.zeros(input_vec.shape)
      num_drop = math.floor(input_vec.shape[0]*self.drop_prob)
      nodes_dropped = []
      for batch in range(input_vec.shape[1]):
        node_dropped = random.choice(np.arange(batch*input_vec.shape[0], (batch+1)*input_vec.shape[0]), num_drop, replace=False)
        nodes_dropped.extend(node_dropped)
      np.put(dropped, nodes_dropped, 1)
      np.place(input_vec, dropped, 0)
      alived = np.where(dropped, 0, 1)
      return input_vec, alived
    elif self.mode == 'inference':
      return input_vec*(1-self.drop_prob)
    else:
      raise ValueError('mode is train or inference.')

  def dense_layer1(self, input_vec):
    return self.dense_layer(input_vec, self.w1, self.b1)

  def sigmoid(self, input_vec):
    return 1/(1+np.exp(-input_vec))

  def dense_layer2(self, input_vec):
    return self.dense_layer(input_vec, self.w2, self.b2)

  def softmax(self, input_vec): # ミニバッチに対応
    return np.exp(input_vec-np.max(input_vec, axis=0)) / (np.sum(np.exp(input_vec-np.max(input_vec, axis=0)), axis=0))

  def im2col(self, input_im):
    pad_wid = ((0, 0), (self.pad_h, self.pad_h), (self.pad_w, self.pad_w))
    pad_im = np.pad(input_im, pad_wid)
    col = np.empty((self.batch_size, self.filter_h, self.filter_w, self.output_y, self.output_x))  # (batch size, filter height, filter width, output height, output width)
    for h in range(self.filter_h):
      for w in range(self.filter_w):
        col[:, h, w, :, :] = pad_im[:, h : h+self.output_y, w : w+self.output_x]
    col = col.transpose(1, 2, 0, 3, 4).reshape(self.filter_w*self.filter_h, self.batch_size*self.output_x*self.output_y)
    return col

  def col2im(self, col):
    col = col.reshape(self.filter_h, self.filter_w, self.batch_size, self.output_y, self.output_x).transpose(2, 0, 1, 3, 4)
    im = np.zeros((self.batch_size, self.im_h+2*self.pad_h, self.im_w+2*self.pad_w))
    for h in range(self.filter_h):
      for w in range(self.filter_w):
        im[:, h : h+self.output_y, w : w+self.output_x] += col[:, h, w, :, :]
    return im

  def col2vec(self, col):
    return col.reshape(self.filter_num, self.batch_size, self.output_y*self.output_x).transpose(0,2,1).reshape(self.filter_num*self.output_y*self.output_x, self.batch_size)

  def convolutional_layer(self, input_mat):
    return self.filter@input_mat+self.filter_bias.reshape(-1,1)

  def postprocessing(self, input_vec): # ミニバッチに対応
    return np.argmax(input_vec, axis=0)

  def cross_entropy(self, true_vec ,pred_vec):
    return np.sum(np.sum(-1 * true_vec * np.log(pred_vec), axis=0)) / self.batch_size

  def accuracy(self, true_y, pred_y): # それぞれのyが1次元の場合にのみ対応済み
    if true_y.shape != pred_y.shape:
      raise ValueError(f'true_y and pred_y must have the same shape. {true_y.shape}, {pred_y.shape}')
    return np.count_nonzero(true_y == pred_y) / true_y.shape[0]

  def train_batch(self, tr_x, tr_y, lr):
    x = self.input_layer(tr_x)
    # t = self.dense_layer1(x)
    col = self.im2col(x)
    t = self.convolutional_layer(col)
    y = self.sigmoid(t)
    # y_dash, alived = self.dropout(y)
    y_dash = self.col2vec(y)  # 全結合層に入れるために1次元配列をbs列並べた形に変形
    a = self.dense_layer2(y_dash)
    y2 = self.softmax(a)
    e_n = self.cross_entropy(tr_y.T, y2)

    # 以下微分の形を行*列で示す
    diff_e_n_with_a = (y2 - tr_y.T)/self.batch_size # クラス数*bs
    diff_e_n_with_y_dash = self.w2.T @ diff_e_n_with_a # units*bs
    diff_e_n_with_w2 = diff_e_n_with_a @ y_dash.T # クラス数*units
    diff_e_n_with_b2 = np.sum(diff_e_n_with_a, axis=1) # クラス数次元ベクトル
    # diff_e_n_with_y = diff_e_n_with_y_dash * alived  # ドロップアウト層の逆伝搬
    diff_e_n_with_y = diff_e_n_with_y_dash.reshape(self.filter_num, self.output_y*self.output_x, self.batch_size).transpose(0,2,1).reshape(self.filter_num, self.batch_size*self.output_x*self.output_y)  # 全結合層から畳み込み層の形に変形
    diff_e_n_with_t = diff_e_n_with_y*(1-y)*y # 
    # diff_e_n_with_x = self.w1.T @ diff_e_n_with_t # 784*bs
    # diff_e_n_with_w1 = diff_e_n_with_t @ x.T # units*784
    # diff_e_n_with_b1 = np.sum(diff_e_n_with_t, axis=1) # units数次元ベクトル
    diff_e_n_with_x = self.col2im(self.filter.T @ diff_e_n_with_t) # 
    diff_e_n_with_filter = diff_e_n_with_t @ col.T # 
    diff_e_n_with_filter_bias = np.sum(diff_e_n_with_t, axis=1) # 

    # 重みの更新
    # self.w2 = self.w2 - lr*diff_e_n_with_w2
    # self.b2 = self.b2 - lr*diff_e_n_with_b2
    self.filter = self.filter - lr*diff_e_n_with_filter
    self.filter_bias = self.filter_bias - lr*diff_e_n_with_filter_bias

    return (self.w1, self.b1, self.w2, self.b2, self.filter, self.filter_bias), e_n
  
  def train(self, train_x, train_y, epochs: int = 10, lr: float = 0.01):
    '''
    train_x: 学習データの特徴量
    train_y: 学習データのラベル
    epochs: エポック数
    lr: 学習率
    '''
    self.batch_size = 100
    history = []
    for i in tqdm(range(epochs)):
      for j in range(60000//self.batch_size):
        tr_x, tr_y = self.preprocessing(train_x, train_y)
        model, entropy = self.train_batch(tr_x, tr_y, lr)
      print(f'Epoch {i+1} end! Cross entroppy is {entropy}.')
      history.append((model,entropy))
    return history

  def predict(self, test_x):
    self.mode = 'inference'
    self.batch_size = 1 # 推論時はバッチ処理を行わない
    # pred_y = [self.postprocessing(self.softmax(self.dense_layer2(self.dropout(self.sigmoid(self.dense_layer1(self.input_layer(np.expand_dims(test_i, axis=0)))))))) for test_i in (test_x)]
    pred_y = [self.postprocessing(self.softmax(self.dense_layer2(self.col2vec(self.sigmoid(self.convolutional_layer(self.im2col(self.input_layer(np.expand_dims(test_i, axis=0))))))))) for test_i in (test_x)]
    pred_y = np.array(pred_y).flatten()
    return pred_y

  def load_best(self, history):
    self.w1, self.b1, self.w2, self.b2, self.filter, self.filter_bias = min(history, key=lambda p: p[1])[0]

  def save_model(self, name):
    np.savez(f'model/{name}', w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2, filter=self.filter, filter_bias=self.filter_bias)

  def load_model(self, name):
    model = np.load(f'model/{name}.npz')
    self.w1, self.b1, self.w2, self.b2, self.filter, self.filter_bias = model['w1'], model['b1'], model['w2'], model['b2'], model['filter'], model['filter_bias']


if __name__ == '__main__': 
  print('please input model name. (ex: 0001)')
  run_name = input()
  logger = Logger()
  train_x = mnist.download_and_parse_mnist_file('train-images-idx3-ubyte.gz')
  train_y = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")
  model = Model(mode ='train', dropout=0.1)
  history = model.train(train_x, train_y, epochs=50, lr=0.1)
  model.load_best(history)
  model.save_model(run_name)
  # model.load_model('0001')

  test_x = mnist.download_and_parse_mnist_file('t10k-images-idx3-ubyte.gz')
  test_y = mnist.download_and_parse_mnist_file('t10k-labels-idx1-ubyte.gz')
  pred_y = model.predict(test_x)
  print(pred_y)

  print('please input log message about this run.')
  logger.info(input())
  logger.info(f'this model is {run_name}')
  logger.info(f'best entropy for train is {min(history, key=lambda p: p[1])[1]}.')
  logger.info(f'accuracy score for test is {model.accuracy(test_y, pred_y)}')
  logger.info('')