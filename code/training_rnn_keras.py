import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/data_only_keras.csv', names = ['x_acc', 'y_acc', 'z_acc'])
label_df = pd.read_csv('data/labels_only_keras.csv', header = None)

traindf = df[ : int(0.7 * len(df))]
testdf = df[int(0.85 * len(df)) : ]
valdf = df[int(0.7 * len(df)) : int(0.85 * len(df))]
trainlabel = label_df[ : int(0.7 * len(df))]
testlabel = label_df[int(0.85 * len(df)) : ]
vallabel = label_df[int(0.7 * len(df)) : int(0.85 * len(df))]

train_gen = TimeseriesGenerator(traindf.values, trainlabel.values, length = 100, batch_size = 16, stride = 100)
test_gen = TimeseriesGenerator(testdf.values, testlabel.values, length = 100, batch_size = 1, stride = 100)
val_gen = TimeseriesGenerator(valdf.values, vallabel.values, length = 100, batch_size = 1, stride = 100)

n_input = 100
n_features = 3
batch_size = 1
model = Sequential()
model.add(keras.layers.LSTM(80, activation = 'relu', input_shape = (n_input, n_features), return_sequences = True))
model.add(keras.layers.LSTM(60, activation = 'relu', return_sequences = True))
model.add(keras.layers.LSTM(40, activation = 'relu'))
model.add(Dense(12, activation = 'softmax'))
adam = keras.optimizers.Adam(lr = 1e-4)
model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = model.fit_generator(train_gen, epochs = 100, verbose = 1)

model.evaluate_generator(test_gen)
model.evaluate_generator(val_gen)

# # Plot training & validation accuracy values
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Val'], loc='upper left')
# plt.show()

# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Val'], loc='upper left')
# plt.show()

