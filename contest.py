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
  model.add(Dense(10, 14*14*32, opt='Adam', opt_kwds={}))
  model.add(Softmax())
  model.load_model('0012')
  Y = model.predict(X, postprocess=False)
  print(Y)

  model2 = Model('inference')
  model2.add(Input((28, 28)))
  model2.add(Conv(input_shape=(28, 28), filter_shape=(5, 5), filter_num=32, opt='Adam', opt_kwds={}))
  model2.add(ReLU())
  model2.add(Pooling(input_shape=(28, 28), channel=32, pool_shape=(2, 2)))
  model2.add(Flatten())
  model2.add(Dropout(dropout=0.4))
  model2.add(Dense(10, 14*14*32, opt='Adam', opt_kwds={}))
  model2.add(Softmax())
  model2.load_model('0016')
  Y2 = model2.predict(X, postprocess=False)
  print(Y2)

  model3 = Model('inference')
  model3.add(Input((28, 28)))
  model3.add(Conv(input_shape=(28, 28), filter_shape=(3, 3), filter_num=32, opt='Adam', opt_kwds={}))
  model3.add(ReLU())
  model3.add(Pooling(input_shape=(28, 28), channel=32, pool_shape=(4, 4)))
  model3.add(Flatten())
  model3.add(Dropout(dropout=0.4))
  model3.add(Dense(10, 7*7*32, opt='Adam', opt_kwds={}))
  model3.add(Softmax())
  model3.load_model('0017')
  Y3 = model3.predict(X, postprocess=False)
  print(Y3)

  Y = (Y+Y2+Y3)/3.0
  Y = model.postprocessing(Y)
  print(Y)
  np.savetxt('predict.txt', Y, fmt='%d')
