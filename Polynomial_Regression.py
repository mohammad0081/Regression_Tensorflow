
import numpy as np
import tensorflow as tf
import random

import matplotlib.pyplot as plt

# Generate the X and y data
X = 4 * np.random.rand(1000) - 2
y = np.array([ 10*i**3 + 3*i + 30 + np.random.randint(-20, 20) for i in X])

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, input_shape=[1], activation='relu'),
    tf.keras.layers.Dense(32 , activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

model.compile(loss='mse', optimizer='adam')
model.fit(X, y, epochs=500)

x_vals = np.linspace(-2, 2, 200)
y_vals = model.predict(x_vals)



plt.scatter(X, y)
plt.plot(x_vals, y_vals , color = 'red')
plt.show()

