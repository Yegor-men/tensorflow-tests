# literally copy pasted from the tutorial. This is just here as a reference

import tensorflow as tf
# imports tensorflow

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# mnist dataset is defined as the mnist dataset from keras

x_train, x_test = x_train / 255.0, x_test / 255.0
# pixel values are normalized between 0 and 1

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  # input layer
  tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU()),
  tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU()),
  tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU()),
  # 3 hidden layers of 256 neurons each with leaky relu activation
  tf.keras.layers.Dense(10)
  # output layer
])
# model architecture is defined

predictions = model(x_train[:1]).numpy()
predictions
# sets the predictions to be the output of the last layer of the model

tf.nn.softmax(predictions).numpy()
# applies softmax to the predictions

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# defines what the loss function is

loss_fn(y_train[:1], predictions).numpy()
# calculates the loss function from the predictions

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
# actually creates the model

model.fit(x_train, y_train, epochs=5)
# trains the model

model.evaluate(x_test,  y_test, verbose=2)
# tests the model

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

probability_model(x_test[:5])