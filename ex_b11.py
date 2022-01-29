import pickle

# import matplotlib.pyplot as plt
# import numpy as np
import tensorflow as tf
# from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow import keras
from tensorflow.keras import datasets, layers, models

from logger import Logger

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(f'{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs.')


print('please input model name. (ex: 0001)')
run_name = 'keras_' + input()
print('please input message about this run.')
run_msg = input()

img_rows, img_cols = 28, 28
num_classes = 10

(X, Y), (Xtest, Ytest) = keras.datasets.mnist.load_data()
X = X.reshape(X.shape[0],img_rows,img_cols,1)
Xtest = Xtest.reshape(Xtest.shape[0],img_rows,img_cols,1)

# 正規化
X = X.astype('float32')/255.0
Xtest = Xtest.astype('float32')/255.0

input_shape = (img_rows,img_cols,1)

Y = keras.utils.to_categorical(Y, num_classes)
Ytest1 = keras.utils.to_categorical(Ytest, num_classes)

model = models.Sequential()
model.add(layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape, padding='same'))
model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Conv2D(128, (5,5), activation='relu', padding='same'))
model.add(layers.Conv2D(128, (5,5), activation='relu', padding='same'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Conv2D(256, (5,5), activation='relu', padding='same'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(num_classes, activation=
'softmax'))

print(model.summary())

model.compile(
  loss=keras.losses.categorical_crossentropy,
  optimizer=keras.optimizers.Adam(),
  metrics=['acc'])

epochs=50
batch_size=32
result = model.fit(X,Y,batch_size=batch_size, epochs=epochs, validation_data=(Xtest, Ytest1))
history = result.history

if run_name != 'keras_':
  model.save(f'model/{run_name}.h5')

  with open(f'history/{run_name}_history.dump', 'wb') as f:
    pickle.dump(history, f)
  
  logger = Logger('keras_general.log')
  logger.info(run_name)
  logger.info(run_msg)
  # TODO: どの履歴を取り出すか要検討
  logger.info(f'loss: {history["loss"][-1]:.5f}')
  logger.info(f'val_loss: {history["val_loss"][-1]:.5f}')
  logger.info(f'acc: {history["acc"][-1]:.3f}')
  logger.info(f'val_acc: {history["val_acc"][-1]:.3f}')
  logger.info('')
else:
  print(f'loss: {history["loss"][-1]}')
  print(f'val_loss: {history["val_loss"][-1]}')
  print(f'acc: {history["acc"][-1]}')
  print(f'val_acc: {history["val_acc"][-1]}')

# fig = plt.figure()
# plt.plot(history['loss'], label='loss')
# plt.plot(history['val_loss'], label='val_loss')
# plt.legend()
# plt.show()

# fig = plt.figure()
# plt.plot(history['acc'], label='acc')
# plt.plot(history['val_acc'], label='val_acc')
# plt.legend()
# plt.show()