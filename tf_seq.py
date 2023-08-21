import tensorflow as tf
device_name = tf.test.gpu_device_name()
print(device_name)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Plot the image
plt.imshow(x_train[0])
plt.show()

# flatten the images
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

model_seq = keras.Sequential(
    [
        keras.Input(shape=(28*28)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ]
)

print(model_seq.summary())

model_seq.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

model_seq.fit(x_train, y_train, batch_size=128, epochs=5, verbose=2)
model_seq.evaluate(x_test, y_test, batch_size=32, verbose=2)


