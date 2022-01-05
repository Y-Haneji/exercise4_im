from ex_advanced import *
import numpy as np

if __name__ == '__main__': 
  X = np.loadtxt("le4MNIST_X.txt")
  X = X.astype('float32')/255.0  # 正規化

  model = Model('inference')
  model.add(Input((28, 28)))
  model.add(Conv(input_shape=(28, 28), filter_shape=(3, 3), filter_num=32, opt='Adam', opt_kwds={}))
  model.add(ReLU())
  model.add(Pooling(input_shape=(28, 28), channel=32, pool_shape=(2, 2)))
  model.add(Flatten())
  model.add(Dropout(dropout=0.4))
  model.add(Dense(10, (32,28,28), opt='Adam', opt_kwds={}))
  model.add(Softmax())
  model.load_model('0012')
  Y = model.predict(X)
  print(Y)
  np.savetxt('predict.txt', Y, fmt='%d')
