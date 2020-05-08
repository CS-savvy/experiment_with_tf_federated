import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow import config
from tensorflow.keras import optimizers, losses, metrics
from matplotlib import pyplot as plt
from pathlib import Path
from models.conv_2 import create_keras_model
from utils import plot_graph

gpus = config.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
        config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


experiment_name = "mnist"
method = "keras_training"
batch_size = 20
epochs = 25
model_name = "conv_simple.h5"
lr = 1e-2

this_dir = Path.cwd()
model_dir = this_dir / "saved_models" / experiment_name / method
output_dir = this_dir / "results" / experiment_name / method

if not model_dir.exists():
    model_dir.mkdir(parents=True)

if not output_dir.exists():
    output_dir.mkdir(parents=True)


# loading mnist dataset

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# preprocessing
x_train = x_train.astype(np.float32).reshape(60000, 28, 28, 1)
y_train = y_train.astype(np.int32).reshape(60000, 1)
x_test = x_test.astype(np.float32).reshape(10000, 28, 28, 1)
y_test = y_test.astype(np.int32).reshape(10000, 1)


normal_model = create_keras_model()
normal_model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                     loss=losses.SparseCategoricalCrossentropy(),
                     metrics=[metrics.SparseCategoricalAccuracy()])

# training
history = normal_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
normal_model.save(model_dir / model_name)


# plotting accuracy and loss graph
fig = plt.figure(figsize=(10, 6))
plot_graph(range(1, len(history.epoch)+1), history.history['sparse_categorical_accuracy'], label='Train Accuracy')
plot_graph(range(1, len(history.epoch)+1), history.history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
plt.legend()
plt.savefig(output_dir / "normal_model_Accuracy.png")


plt.figure(figsize=(10, 6))
plot_graph(range(1, len(history.epoch)+1), history.history['loss'], label='Train loss')
plot_graph(range(1, len(history.epoch)+1), history.history['val_loss'], label='Validation loss')
plt.legend()
plt.savefig(output_dir / "normal_model_loss.png")