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

class SGD:
  def __init__(self, lr: float=0.01):
    self.lr = lr
  
  def update(self, grad_w, grad_b):
    dw = -self.lr*grad_w
    db = -self.lr*grad_b
    return dw, db


def get_opt(opt: str, *args, **kwds):
  if opt == 'SGD':
    optimizer = SGD(*args, **kwds)
  else:
    raise ValueError(f'{opt} is not implemented.')

  return optimizer

class Layer:
  def __init__(self, *args, **kwds):
    pass

  def forward(self, x, *args, **kwds):
    self.x = x
    self.y = x
    return self.y

  def backward(self, grad, *args, **kwds):
    self.grad_x = grad
    return self.grad_x

  def update(self, *args, **kwds):
    pass

  def get_weight(self, *args, **kwds) -> tuple:
    return {}

  def load_weight(self, weight_dic):
    pass


class Input(Layer):
  def __init__(self, output_shape: tuple):
    self.output_shape = output_shape
  
  def forward(self, x, batch_size=100, mode='train'):
    self.x = x
    self.y = x.reshape(((batch_size,) + self.output_shape)).T
    return self.y


class Dense(Layer):
  def __init__(self, units: int, input_shape: tuple, name: str='dense', opt: str='SGD', opt_kwds: dict={}):
    self.units = units
    self.input_shape = input_shape
    input_size = 1
    for size in input_shape:
      input_size *= size

    self.w = random.normal(loc=0, scale=np.sqrt(1/input_size), size=input_size*units).reshape(units, input_size)
    self.b = random.normal(loc=0, scale=np.sqrt(1/input_size), size=units)
    self.name = name
    self.opt = get_opt(opt, **opt_kwds)

  def forward(self, x, batch_size=100, mode='train'): # ミニバッチに対応
    # TODO: 画像入力に対応
    self.x = x
    self.y = self.w@x + self.b.reshape(-1, 1)
    return self.y

  def backward(self, grad):
    self.grad_x = self.w.T@grad
    self.grad_w = grad@self.x.T
    self.grad_b = np.sum(grad, axis=1)
    return self.grad_x

  def update(self):
    dw, db = self.opt.update(self.grad_w, self.grad_b)
    self.w += dw
    self.b += db

  def get_weight(self) -> tuple:
    return {f'{self.name}_w': self.w, f'{self.name}_b': self.b}

  def load_weight(self, weight_dic):
    self.w = weight_dic[f'{self.name}_w']
    self.b = weight_dic[f'{self.name}_b']

class Dropout(Layer):
  def __init__(self, dropout: float=0.1):
    self.dropout = dropout
  
  def forward(self, x, batch_size=100, mode='train'):
    self.x = x
    if mode == 'train':
      dropped = np.zeros(x.shape)
      num_drop = math.floor(x.shape[0]*self.dropout)
      nodes_dropped = []
      for batch in range(x.shape[1]):
        node_dropped = random.choice(np.arange(batch*x.shape[0], (batch+1)*x.shape[0]), num_drop, replace=False)
        nodes_dropped.extend(node_dropped)

      np.put(dropped, nodes_dropped, 1)
      self.y = x
      np.place(self.y, dropped, 0)
      self.flag_not_dropped = np.where(dropped, 0, 1)
      return self.y

    if mode == 'inference':
      return x*(1-self.dropout)

  def backward(self, grad):
    self.grad_x = grad*self.flag_not_dropped
    return self.grad_x


class Sigmoid(Layer):
  def __init__(self):
    pass

  def forward(self, x, batch_size=100, mode='train'):
    sigmoid_range = 34.538776394910684
    x = np.clip(x, -sigmoid_range, sigmoid_range)
    self.x = x
    self.y = 1/(1.0+np.exp(-x))
    return self.y

  def backward(self, grad):
    self.grad_x = grad*(1-self.y)*self.y
    return self.grad_x


class ReLU(Layer):
  def __init__(self):
    pass

  def forward(self, x, batch_size=100, mode='train'):
    self.x = x
    self.y = np.where(x>0, x, 0)
    return self.y

  def backward(self, grad):
    self.grad_x = grad*np.where(self.x>0, 1, 0)
    return self.grad_x


class Softmax(Layer):
  def __init__(self):
    pass

  def forward(self, x, batch_size=100, mode='train'):
    self.batch_size = batch_size
    self.x = x
    self.y = np.exp(x-np.max(x, axis=0)) / (np.sum(np.exp(x-np.max(x, axis=0)), axis=0))
    return self.y

  def backward(self, true_y):
    grad_x = (self.y - true_y.T)/self.batch_size
    return grad_x


class Model:
  def __init__(self, mode = 'train') -> None:
    # モデルのアーキテクチャを作成
    self.batch_size = 100
    if mode not in ['train', 'inference']:
      raise ValueError('mode is train or inference.')
    self.mode = mode
    self.layers = []
    self.weights_dic = {}

  def add(self, layer: Layer):
    self.layers.append(layer)
    self.weights_dic.update(layer.get_weight())


  def preprocessing(self, train_x, train_y):
    '''学習用セットからバッチを作成する'''
    idx = random.randint(0, len(train_y), self.batch_size)
    tr_x = train_x[idx]
    l = [[1 if i == label else 0 for i in range(10)] for label in train_y[idx]]
    tr_y = np.zeros((len(l), len(l[0])))
    tr_y[:] = l
    return tr_x, tr_y # tr_x is (bs, 28, 28) tr_y is one-hot-encoding (bs, 10)

  def postprocessing(self, input_vec): # ミニバッチに対応
    return np.argmax(input_vec, axis=0)

  def cross_entropy(self, true_vec ,pred_vec):
    return np.sum(np.sum(-1 * true_vec * np.log(pred_vec), axis=0)) / self.batch_size

  def accuracy(self, true_y, pred_y): # それぞれのyが1次元の場合にのみ対応済み
    if true_y.shape != pred_y.shape:
      raise ValueError(f'true_y and pred_y must have the same shape. {true_y.shape}, {pred_y.shape}')
    return np.count_nonzero(true_y == pred_y) / true_y.shape[0]

  def train_batch(self, tr_x, tr_y, lr):
    flag_input = True
    for layer in self.layers:
      if flag_input == True:
        x = tr_x
        flag_input = False
      x = layer.forward(x, self.batch_size, self.mode)

    entropy = self.cross_entropy(tr_y, x.T)

    flag_output = True
    for layer in reversed(self.layers):
      if flag_output == True:
        grad = tr_y
        flag_output = False
      grad = layer.backward(grad)

    for layer in self.layers:
      layer.update()

    weights_dic = {}
    for layer in self.layers:
      weights_dic.update(layer.get_weight())
    return weights_dic, entropy
  
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
      entropies = []
      for j in range(60000//self.batch_size):
        tr_x, tr_y = self.preprocessing(train_x, train_y)
        self.weights_dic, entropy = self.train_batch(tr_x, tr_y, lr)
        entropies.append(entropy)
      entropy = sum(entropies)/len(entropies)
      print(f'Epoch {i+1} end! Cross entroppy is {entropy}.')
      history.append((self.weights_dic,entropy))
    return history

  def predict(self, test_x):
    self.mode = 'inference'
    self.batch_size = 1 # 推論時はバッチ処理を行わない
    pred_y = []
    for test_i in (test_x):
      flag_input = True
      for layer in self.layers:
        if flag_input == True:
          x = test_i
          flag_input = False
        x = layer.forward(x, self.batch_size, self.mode)
      pred_y.append(self.postprocessing(self.layers[-1].y))

    pred_y = np.array(pred_y).flatten()
    return pred_y

  def load_best(self, history):
    self.weights_dic = history[np.nanargmin(list(zip(*history))[1])][0]
    for layer in self.layers:
      layer.load_weight(self.weights_dic)

  def save_model(self, name):
    np.savez(f'model/{name}', **self.weights_dic)

  def load_model(self, name):
    self.weights_dic = np.load(f'model/{name}.npz')
    for layer in self.layers:
      layer.load_weight(self.weights_dic)

if __name__ == '__main__': 
  print('please input model name. (ex: 0001)')
  run_name = input()
  logger = Logger()
  train_x = mnist.download_and_parse_mnist_file('train-images-idx3-ubyte.gz')
  train_y = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")

  model = Model(mode ='train')
  model.add(Input((28*28,)))
  model.add(Dense(32, (28*28,), 'SGD', {'lr': 0.01}))
  model.add(ReLU())
  model.add(Dropout(dropout=0.1))
  model.add(Dense(10, (32,), name='dense2', opt='SGD', opt_kwds={'lr': 0.01}))
  model.add(Softmax())

  model.load_model('0003')
  history = model.train(train_x, train_y)
  model.load_best(history)
  model.save_model(run_name)

  test_x = mnist.download_and_parse_mnist_file('t10k-images-idx3-ubyte.gz')
  test_y = mnist.download_and_parse_mnist_file('t10k-labels-idx1-ubyte.gz')
  pred_y = model.predict(test_x)
  print(pred_y)

  print('please input log message about this run.')
  logger.info(input())
  logger.info(f'this model is {run_name}')
  logger.info(f'best entropy for train is {np.nanmin(list(zip(*history))[1])}.')
  logger.info(f'accuracy score for test is {model.accuracy(test_y, pred_y)}')
  logger.info('')