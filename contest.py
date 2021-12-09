from ex_advanced import Model
import mnist
import numpy as np

if __name__ == '__main__': 
  X = np.loadtxt("le4MNIST_X.txt")

  model = Model('inference')
  model.load_model('0002')
  Y = model.predict(X)
  np.savetxt('predict.txt', Y, fmt='%d')
