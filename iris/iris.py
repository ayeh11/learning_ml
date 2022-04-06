import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

training_size = 120
dataframe = pd.read_csv("iris/iris.csv", header=None)
data = tf.random.shuffle(dataframe.values)
X_train = data[0:training_size, 0:4]
X_test = data[training_size:, 0:4]
y_train = data[0:training_size,4]
y_test = data[training_size:,4]

#Configure the model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(64, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(32, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=10)

results = model.evaluate(X_test, y_test)
print(f'Test loss: {results[0]} / Test accuracy: {results[1]}')

actual = y_test
print(actual)
predictions = model.predict(X_test[:2])
print(np.argmax(predictions, axis=1))
