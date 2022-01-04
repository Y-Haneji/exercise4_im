import numpy as np
from ex_advanced import Pooling, ReLU

pool = Pooling(input_shape=(4,4), channel=1, pool_shape=(2,2))
x = np.array([[[[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]]]])
y = pool.forward(x, batch_size=1)
grad = np.ones((2,2))
grad_x = pool.backward(grad)
print(x)
print(y)
print(grad_x)