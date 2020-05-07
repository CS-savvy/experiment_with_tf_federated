import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

def plot_graph(X, y, format = '-', label=''):
    plt.plot(X, y, format, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(True)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype(np.float32).reshape(60000, 28, 28, 1)
y_train = y_train.astype(np.int32).reshape(60000, 1)
x_test = x_test.astype(np.float32).reshape(10000, 28, 28, 1)
y_test = y_test.astype(np.int32).reshape(10000, 1)

print()
def create_keras_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

model = create_keras_model()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

history = model.fit(x_train, y_train, batch_size=20, epochs=5, validation_data=(x_test, y_test))


fig = plt.figure(figsize=(10, 6))
plot_graph(range(1, len(history.epoch)+1), history.history['sparse_categorical_accuracy'], label='Train Accuracy')
plot_graph(range(1, len(history.epoch)+1), history.history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
plt.legend()
plt.savefig("Accuracy.png")

plt.figure(figsize=(10, 6))
plot_graph(range(1, len(history.epoch)+1), history.history['loss'], label='Train loss')
plot_graph(range(1, len(history.epoch)+1), history.history['val_loss'], label='Validation loss')
plt.legend()
plt.savefig("loss.png")