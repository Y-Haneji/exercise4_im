import math
import pickle
import sys

import matplotlib.pyplot as plt
import mnist
import numpy as np
from numpy import random
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from logger import Logger

random.seed(71)

class SGD:
  def __init__(self, lr: float=0.01):
    self.lr = lr
  
  def update(self, grad_w, grad_b):
    dw = -self.lr*grad_w
    db = -self.lr*grad_b
    return dw, db


class MomentumSGD:
  def __init__(self, lr: float=0.01, alpha: float=0.9):
    self.lr = lr
    self.alpha = alpha

  def update(self, grad_w, grad_b):
    self.dw = self.alpha*self.dw - self.lr*grad_w
    self.db = self.alpha*self.db - self.lr*grad_b
    return self.dw, self.db


class AdaGrad:
  def __init__(self, lr: float=0.001, h0: float=1e-8):
    self.lr = lr
    self.h_w = h0
    self.h_b = h0

  def update(self, grad_w, grad_b):
    self.h_w = self.h_w + grad_w*grad_w
    self.h_b = self.h_b + grad_b*grad_b
    dw = -self.lr/np.sqrt(self.h_w)*grad_w
    db = -self.lr/np.sqrt(self.h_b)*grad_b
    return dw, db


class RMSProp:
  def __init__(self, lr: float=0.001, rho: float=0.9, epsilon: float=1e-8, h0: float=1e-8):
    self.lr = lr
    self.rho = rho
    self.epsilon = epsilon
    self.h_w = h0
    self.h_b = h0

  def update(self, grad_w, grad_b):
    self.h_w = self.rho*self.h_w + (1-self.rho)*grad_w*grad_w
    self.h_b = self.rho*self.h_b + (1-self.rho)*grad_b*grad_b
    dw = -self.lr/np.sqrt(self.h_w)*grad_w
    db = -self.lr/np.sqrt(self.h_b)*grad_b
    return dw, db


class AdaDelta:
  def __init__(self, rho: float=0.95, epsilon: float=1e-6):
    self.rho = rho
    self.epsilon = epsilon
    self.h_w = 0
    self.h_b = 0
    self.s_w = 0
    self.s_b = 0

  def update(self, grad_w, grad_b):
    self.h_w = self.rho*self.h_w + (1-self.rho)*grad_w*grad_w
    self.h_b = self.rho*self.h_b + (1-self.rho)*grad_b*grad_b
    dw = -np.sqrt(self.s_w+self.epsilon)/np.sqrt(self.h_w+self.epsilon)*grad_w
    db = -np.sqrt(self.s_b+self.epsilon)/np.sqrt(self.h_b+self.epsilon)*grad_b
    self.s_w = self.rho*self.s_w + (1-self.rho)*dw*dw
    self.s_b = self.rho*self.s_b + (1-self.rho)*db*db
    return dw, db


class Adam:
  def __init__(self, alpha: float=0.001, beta_1: float=0.9, beta_2: float=0.999, epsilon: float=1e-8):
    self.alpha = alpha
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.t = 0
    self.m_w = 0
    self.m_b = 0
    self.v_w = 0
    self.v_b = 0

  def update(self, grad_w, grad_b):
    self.t = self.t+1
    self.m_w = self.beta_1*self.m_w + (1-self.beta_1)*grad_w
    self.m_b = self.beta_1*self.m_b + (1-self.beta_1)*grad_b
    self.v_w = self.beta_2*self.v_w + (1-self.beta_2)*grad_w*grad_w
    self.v_b = self.beta_2*self.v_b + (1-self.beta_2)*grad_b*grad_b
    self.m_w_hat = self.m_w/(1-np.power(self.beta_1, self.t))
    self.m_b_hat = self.m_b/(1-np.power(self.beta_1, self.t))
    self.v_w_hat = self.v_w/(1-np.power(self.beta_2, self.t))
    self.v_b_hat = self.v_b/(1-np.power(self.beta_2, self.t))
    dw = -self.alpha*self.m_w_hat/(np.sqrt(self.v_w_hat)+self.epsilon)
    db = -self.alpha*self.m_b_hat/(np.sqrt(self.v_b_hat)+self.epsilon)
    return dw, db


def get_opt(opt: str, *args, **kwds):
  opt_dic = {
    'SGD': SGD,
    'MomentumSGD': MomentumSGD,
    'AdaGrad': AdaGrad,
    'RMSProp': RMSProp,
    'AdaDelta': AdaDelta,
    'Adam': Adam
  }

  if opt in opt_dic:
    optimizer = opt_dic[opt](*args, **kwds)
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
    self.output_dim = len(output_shape)
  
  def forward(self, x, batch_size=100, mode='train'):
    self.x = x
    if self.output_dim == 1:
      self.y = x.reshape((batch_size,) + self.output_shape).T
    else:  # 画像データをそのまま扱う場合を想定
      self.y = x.reshape((batch_size,) + self.output_shape)
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

  def forward(self, x, batch_size=100, mode='train'):
    self.x = x
    self.batch_size = batch_size
    self.y = self.w@self.x + self.b.reshape(-1, 1)
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


class Conv(Layer):
  def __init__(self, *, input_shape: tuple, filter_shape: tuple, filter_num: int, name: str='conv', opt: str='SGD', opt_kwds: dict={}):
    self.im_h = input_shape[0]
    self.im_w = input_shape[1]
    self.filter_h = filter_shape[0]
    self.filter_w = filter_shape[1]
    self.pad_h = math.floor(self.filter_h/2)
    self.pad_w = math.floor(self.filter_w/2)
    self.o_h = self.im_h-self.filter_h+2*self.pad_h+1
    self.o_w = self.im_w-self.filter_w+2*self.pad_w+1
    self.filter_num = filter_num
    self.w = 0.05*random.randn(self.filter_num, self.filter_h*self.filter_w)
    self.b = 0.05*random.randn(self.filter_num)
    self.name = name
    self.opt = get_opt(opt, **opt_kwds)

  def _im2col(self, im, batch_size):
    pad_wid = ((0, 0), (self.pad_h, self.pad_h), (self.pad_w, self.pad_w))
    pad_im = np.pad(im, pad_wid)
    col = np.empty((batch_size, self.filter_h, self.filter_w, self.o_h, self.o_w))  # (batch size, filter height, filter width, output height, output width)
    for h in range(self.filter_h):
      for w in range(self.filter_w):
        col[:, h, w, :, :] = pad_im[:, h : h+self.o_h, w : w+self.o_w]
    col = col.transpose(1, 2, 0, 3, 4).reshape(self.filter_w*self.filter_h, batch_size*self.o_w*self.o_h)
    return col

  def _col2im(self, col, batch_size):
    col = col.reshape(self.filter_h, self.filter_w, batch_size, self.o_h, self.o_w).transpose(2, 0, 1, 3, 4)
    im = np.zeros((batch_size, self.im_h+2*self.pad_h, self.im_w+2*self.pad_w))
    for h in range(self.filter_h):
      for w in range(self.filter_w):
        im[:, h : h+self.o_h, w : w+self.o_w] += col[:, h, w, :, :]
    return im

  def forward(self, x, batch_size=100, mode='train'):
    self.x = self._im2col(x, batch_size)
    self.batch_size = batch_size
    self.y = self.w@self.x + self.b.reshape(-1, 1)
    self.y = self.y.reshape(self.filter_num, batch_size, self.o_h, self.o_w).transpose(1, 0, 2, 3)
    return self.y

  def backward(self, grad):
    grad = grad.transpose(1, 0, 2, 3).reshape(self.filter_num, self.batch_size*self.o_h*self.o_w)
    self.grad_x = self._col2im(self.w.T@grad, self.batch_size)
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


class Pooling(Layer):
  def __init__(self, *, input_shape: tuple, channel: int, pool_shape: tuple):
    self.im_h = input_shape[0]
    self.im_w = input_shape[1]
    self.channel = channel
    self.pool_h = pool_shape[0]
    self.pool_w = pool_shape[1]
    self.o_h = self.im_h//self.pool_h
    self.o_w = self.im_w//self.pool_w

  def _im2col(self, im, batch_size):
    col = np.empty((batch_size, self.channel, self.pool_h, self.pool_w, self.o_h, self.o_w))
    for h in range(self.pool_h):
      for w in range(self.pool_w):
        col[:, :, h, w, :, :] = im[:, :, h : h+self.pool_h*self.o_h : self.pool_h, w : w+self.pool_w*self.o_w : self.pool_w]
    col = col.transpose(0, 1, 4, 5, 2, 3).reshape(batch_size*self.channel*self.o_h*self.o_w, self.pool_h*self.pool_w)
    return col

  def _col2im(self, col, batch_size):
    col = col.reshape(self.batch_size, self.channel, self.o_h, self.o_w, self.pool_h, self.pool_w).transpose(0, 1, 4, 5, 2, 3)
    im = np.zeros((batch_size, self.channel, self.im_h, self.im_w))
    for h in range(self.pool_h):
      for w in range(self.pool_w):
        im[:, :, h : h+self.pool_h*self.o_h : self.pool_h, w : w+self.pool_w*self.o_w : self.pool_w] += col[:, :, h, w, :, :]
    return im

  def forward(self, x, batch_size=100, mode='train'):
    self.x = self._im2col(x, batch_size)
    self.batch_size = batch_size
    self.ind = np.argmax(self.x, axis=1)
    self.y = np.max(self.x, axis=1).reshape((batch_size, self.channel, self.o_h, self.o_w))
    return self.y

  def backward(self, grad):
    self.grad_x = np.zeros((self.batch_size*self.channel*self.o_h*self.o_w, self.pool_h*self.pool_w))
    self.grad_x[np.arange(self.ind.size), self.ind] = grad.flatten()
    self.grad_x = self._col2im(self.grad_x, self.batch_size)
    return self.grad_x


class Flatten(Layer):
  def forward(self, x, batch_size=100, mode='train'):
    self.x = x
    self.batch_size = batch_size
    self.y = x.reshape((batch_size, -1)).T
    return self.y
  
  def backward(self, grad):
    self.grad_x = grad.T.reshape((self.batch_size, self.x.shape[1:]))
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


class BatchNormalization(Layer):
  def __init__(self, units: int=96, name: str='bn', opt: str='SGD', opt_kwds: dict={}):
    self.name = name
    self.opt = get_opt(opt, **opt_kwds)
    self.gamma = np.ones((units,1))
    self.beta = np.zeros((units,1))
    self.mean_list = []
    self.variance_list = []

  def forward(self, x, batch_size=100, mode='train'):
    if mode == 'train':
      self.x = x
      self.batch_size = batch_size
      self.mean = np.expand_dims(np.sum(x, axis=1)/self.batch_size, axis=-1)
      self.variance = np.expand_dims(np.sum(np.square(x-self.mean), axis=1)/self.batch_size, axis=-1)
      self.x_normalized = (x-self.mean)/np.sqrt(self.variance+sys.float_info.epsilon)
      self.y = self.gamma*self.x_normalized + self.beta
      self.mean_list.append(self.mean)
      self.variance_list.append(self.variance)
      self.mean_expected = np.average(np.array(self.mean_list), axis=0)
      self.variance_expected = np.average(np.array(self.variance_list), axis=0)
      return self.y
    elif mode == 'inference':
      return self.gamma/np.sqrt(self.variance_expected+sys.float_info.epsilon)*x + (self.beta - self.gamma*self.mean_expected/np.sqrt(self.variance_expected+sys.float_info.epsilon))
    else:
      raise ValueError('mode is train or inference.')

  def backward(self, grad):
    self.grad_x_normalized = grad*self.gamma
    self.grad_variance = np.expand_dims(np.sum(grad*(self.x-self.mean)*(-1)/2*np.power(self.variance+sys.float_info.epsilon, -3/2), axis=1), axis=-1)
    self.grad_mean = np.expand_dims(np.sum(self.grad_x_normalized*(-1)/np.sqrt(self.variance+sys.float_info.epsilon), axis=1), axis=-1) + self.grad_variance*np.expand_dims(np.sum(-2*(self.x-self.mean), axis=1)/self.batch_size, axis=-1)
    self.grad_x = self.grad_x_normalized/np.sqrt(self.variance+sys.float_info.epsilon) + self.grad_variance*2*(self.x-self.mean)/self.batch_size + self.grad_mean/self.batch_size
    self.grad_gamma = np.expand_dims(np.sum(grad*self.x_normalized, axis=1), axis=-1)
    self.grad_beta = np.expand_dims(np.sum(grad, axis=1), axis=-1)
    return self.grad_x

  def update(self):
    dgamma, dbeta = self.opt.update(self.grad_gamma, self.grad_beta)
    self.gamma += dgamma
    self.beta += dbeta

  def get_weight(self) -> tuple:
    return {f'{self.name}_gamma': self.gamma, f'{self.name}_beta': self.beta}

  def load_weight(self, weight_dic):
    self.gamma = weight_dic[f'{self.name}_gamma']
    self.beta = weight_dic[f'{self.name}_beta']


class Dropout(Layer):
  def __init__(self, dropout: float=0.1):
    self.dropout = dropout
  
  def forward(self, x, batch_size=100, mode='train'):
    self.x = x
    if mode == 'train':
      self.mask = np.random.rand(*self.x.shape) >= self.dropout
      self.y = x*self.mask
      return self.y
    elif mode == 'inference':
      return x*(1-self.dropout)
    else:
      raise ValueError('mode is train or inference.')

  def backward(self, grad):
    self.grad_x = grad*self.mask
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


def to_one_hot_vector(labels):
  l = [[1 if i == label else 0 for i in range(10)] for label in labels]
  ohv = np.zeros((len(l), len(l[0])))
  ohv[:] = l
  return ohv

def accuracy(true_y, pred_y):  # ラベル表示に対応、one-hot vectorに非対応
  if true_y.shape != pred_y.shape:
    raise ValueError(f'true_y and pred_y must have the same shape. {true_y.shape}, {pred_y.shape}')
  return np.count_nonzero(true_y == pred_y) / true_y.shape[0]

def cross_entropy(true_vec ,pred_vec):  # one-hot vectorに対応
  return np.sum(np.sum(-1 * true_vec * np.log(pred_vec), axis=1)) / true_vec.shape[0]


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
    '''学習データセットからバッチを作成する'''
    idx = random.randint(0, len(train_y), self.batch_size)
    tr_x = train_x[idx]
    l = [[1 if i == label else 0 for i in range(10)] for label in train_y[idx]]
    tr_y = np.zeros((len(l), len(l[0])))
    tr_y[:] = l
    return tr_x, tr_y # tr_x is (bs, 28, 28) tr_y is one-hot-encoding (bs, 10)

  def postprocessing(self, input_vec):  # ミニバッチに対応、bs*class
    return np.argmax(input_vec, axis=1)

  def valid_batch(self, val_x, val_y, lr):
    self.mode = 'inference'
    flag_input = True
    for layer in self.layers:
      if flag_input == True:
        x = val_x
        flag_input = False
      x = layer.forward(x, self.batch_size, self.mode)

    entropy = cross_entropy(val_y, x.T)
    return entropy

  def valid(self, valid_x, valid_y, lr: float):
    entropies = []
    for i in range(valid_y.shape[0]//self.batch_size):
      val_x, val_y = self.preprocessing(valid_x, valid_y)
      entropy = self.valid_batch(val_x, val_y, lr)
      entropies.append(entropy)
    entropy = sum(entropies)/len(entropies)
    return entropy

  def train_batch(self, tr_x, tr_y, lr):
    self.mode = 'train'
    flag_input = True
    for layer in self.layers:
      if flag_input == True:
        x = tr_x
        flag_input = False
      x = layer.forward(x, self.batch_size, self.mode)

    entropy = cross_entropy(tr_y, x.T)

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
  
  def train(self, train_x, train_y, valid_x=None, valid_y=None, valid: int=0, epochs: int = 10, lr: float = 0.01):
    '''
    train_x: 学習データの特徴量
    train_y: 学習データのラベル
    valid_x: バリデーションデータの特徴量
    valid_y: バリデーションデータのラベル
    valid: バリデーションの間隔(0: バリデーションしない)
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

      if valid == 1:
        val_entropy = self.valid(valid_x, valid_y, lr)
        print(f'Epoch {i+1} end! loss: {entropy:.5f}, val_loss: {val_entropy:.5f}')
        history.append((self.weights_dic, entropy, val_entropy))
      elif valid >= 2:
        if i%valid == 0:
          val_entropy = self.valid(valid_x, valid_y, lr)
          print(f'Epoch {i+1} end! loss: {entropy:.5f}, val_loss: {val_entropy:.5f}')
        else:
          print(f'Epoch {i+1} end! loss: {entropy:.5f}.')
        history.append((self.weight_dic, entropy))
      else:
        print(f'Epoch {i+1} end! loss: {entropy:.5f}.')
        history.append((self.weights_dic, entropy))
    
    history = dict(zip(['weight', 'loss', 'val_loss'], list(zip(*history))))
    return history

  def predict(self, test_x, test_y=None, valid=False):
    '''
    train_x: テストデータの特徴量
    test_y(option): テストデータのラベル
    valid: テストデータのlossを計算するかどうかのフラグ
    '''
    self.mode = 'inference'
    self.batch_size = 1 # 推論時はバッチ処理を行わない
    if valid == False:
      pred_y = []
      for test_i in (test_x):
        flag_input = True
        for layer in self.layers:
          if flag_input == True:
            x = test_i
            flag_input = False
          x = layer.forward(x, self.batch_size, self.mode)
        pred_y.append(self.layers[-1].y)
      pred_y = self.postprocessing(np.array(pred_y))
      return pred_y
    elif valid == True:
      test_y = to_one_hot_vector(test_y)
      pred_y = []
      for test_i in (test_x):
        flag_input = True
        for layer in self.layers:
          if flag_input == True:
            x = test_i
            flag_input = False
          x = layer.forward(x, self.batch_size, self.mode)
        pred_y.append(self.layers[-1].y.flatten())
      pred_y = np.array(pred_y)
      entropy = cross_entropy(test_y, pred_y)
      pred_y = self.postprocessing(pred_y)
      return pred_y, entropy
    else:
      raise ValueError(f'please set bool not {valid} for valid.')

  def load_best(self, history):
    if 'val_loss' in history:
      self.weights_dic = history['weight'][np.nanargmin(history['val_loss'])]
    else:
      self.weight_dic = history['weight'][np.nanargmin(history['loss'])]
    for layer in self.layers:
      layer.load_weight(self.weights_dic)

  def save_model(self, name):
    np.savez(f'model/{name}', **self.weights_dic)

  def load_model(self, name):
    print(f'loading {name}...')
    self.weights_dic = np.load(f'model/{name}.npz')
    for layer in self.layers:
      layer.load_weight(self.weights_dic)

  def save_history(self, name, history):
    with open(f'history/{name}_history.dump', 'wb') as f:
      pickle.dump(history, f)


if __name__ == '__main__': 
  print('please input model name. (ex: 0001)')
  run_name = input()
  print('please input message about this run.')
  run_msg = input()

  logger = Logger()
  train_x = mnist.download_and_parse_mnist_file('train-images-idx3-ubyte.gz')
  train_y = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")
  test_x = mnist.download_and_parse_mnist_file('t10k-images-idx3-ubyte.gz')
  test_y = mnist.download_and_parse_mnist_file('t10k-labels-idx1-ubyte.gz')
  # 正規化
  train_x = train_x.astype('float32')/255.0
  test_x = test_x.astype('float32')/255.0

  model = Model(mode='train')
  # model.add(Input((28*28,)))
  # model.add(Dense(96, (28*28,), name='dense1', opt='Adam', opt_kwds={}))
  # model.add(Sigmoid())
  # model.add(ReLU())
  # model.add(BatchNormalization(96))
  # model.add(Dropout(dropout=0.1))
  # model.add(Dense(10, (96,), name='dense2', opt='Adam', opt_kwds={}))
  model.add(Input((28, 28)))
  model.add(Conv(input_shape=(28, 28), filter_shape=(3, 3), filter_num=32, opt='SGD', opt_kwds={}))
  model.add(Sigmoid())
  model.add(Dropout(0.4))
  # model.add(Pooling(input_shape=(28, 28), channel=32, pool_shape=(2, 2)))
  # model.add(Dropout(0.4))
  model.add(Dense(10, (32,28,28), opt='SGD', opt_kwds={}))
  model.add(Softmax())

  # model.load_model('0007')
  history = model.train(train_x, train_y, test_x, test_y, valid=1, epochs=100)
  model.save_history(run_name, history)
  model.load_best(history)
  model.save_model(run_name)

  pred_y, val_entropy = model.predict(test_x, test_y, valid=True)
  print(pred_y)

  if run_name != '':
    logger.info(run_name)
    logger.info(run_msg)
    logger.info(f'loss: {np.nanmin(history["loss"])}.')
    logger.info(f'val_loss: {val_entropy}')
    logger.info(f'val_acc: {accuracy(test_y, pred_y)}')
    logger.info('')
  else:
    print(f'loss: {np.nanmin(history["loss"])}.')
    print(f'val_loss {val_entropy}')
    print(f'val_acc {model.accuracy(test_y, pred_y)}')
