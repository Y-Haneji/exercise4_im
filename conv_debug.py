from ex_advanced import *

tr_x = np.array([[[0,1,2],[3,4,5],[6,7,8]]])
tr_y = np.array([[1,0,0,0,0,0,0,0,0,0]])

model = Model()

model.train_batch(tr_x, tr_y, 0.1)