import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from matplotlib import pyplot as plt
from pathlib import Path

this_dir = Path.cwd()
model_dir = this_dir / "models"
output_dir = this_dir / "results"

if not model_dir.exists():
    model_dir.mkdir()

if not output_dir.exists():
    output_dir.mkdir()


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

x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.int32)
x_test = x_test.astype(np.float32).reshape(10000, 28, 28, 1)
y_test = y_test.astype(np.int32).reshape(10000, 1)

total_image_count = len(x_train)
split = 4
image_per_set = int(np.floor(total_image_count/split))

client_train_dataset = collections.OrderedDict()
for i in range(1, split+1):
    client_name = "client_" + str(i)
    start = image_per_set * (i-1)
    end = image_per_set * i

    print(f"Adding data from {start} to {end} for client : {client_name}")
    data = collections.OrderedDict((('label', y_train[start:end]), ('pixels', x_train[start:end])))
    client_train_dataset[client_name] = data

train_dataset = tff.simulation.FromTensorSlicesClientData(client_train_dataset)

example_dataset = train_dataset.create_tf_dataset_for_client(train_dataset.client_ids[0])
example_element = next(iter(example_dataset))

NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = image_per_set
PREFETCH_BUFFER = 10


def preprocess(dataset):

  def batch_format_fn(element):
    """Flatten a batch `pixels` and return the features as an `OrderedDict`."""

    return collections.OrderedDict(
        x=tf.reshape(element['pixels'], [-1, 28, 28, 1]),
        y=tf.reshape(element['label'], [-1, 1]))

  return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(
      BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)


preprocessed_example_dataset = preprocess(example_dataset)
sample_batch = tf.nest.map_structure(lambda x: x.numpy(), next(iter(preprocessed_example_dataset)))

#print("Sample from dataset pipeline :", sample_batch, "\n\n", sample_batch['x'].shape, sample_batch['y'].shape)


def make_federated_data(client_data, client_ids):
    return [preprocess(client_data.create_tf_dataset_for_client(x)) for x in client_ids]


federated_train_data = make_federated_data(train_dataset, train_dataset.client_ids)

print('Number of client datasets: {l}'.format(l=len(federated_train_data)))
print('First dataset: {d}'.format(d=federated_train_data[0]))


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


x_train = x_train.reshape(60000, 28, 28, 1)
y_train = y_train.reshape(60000, 1)
normal_model = create_keras_model()
normal_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

history = normal_model.fit(x_train, y_train, batch_size=20, epochs=25, validation_data=(x_test, y_test))
normal_model.save(model_dir / "without_tff_model.h5")


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


def model_fn():
  # We _must_ create a new model here, and _not_ capture it from an external
  # scope. TFF will call this within different graph contexts.

  keras_model = create_keras_model()
  return tff.learning.from_keras_model(
      keras_model,
      input_spec=preprocessed_example_dataset.element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.01),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.01))

print(str(iterative_process.initialize.type_signature))

state = iterative_process.initialize()

NUM_ROUNDS = 5
tff_train_acc = []
tff_val_acc = []
tff_train_loss = []
tff_val_loss = []
for round_num in range(1, NUM_ROUNDS+1):
    state, metrics = iterative_process.next(state, federated_train_data)
    eval_model = create_keras_model()
    eval_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                       loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                       metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    tff.learning.assign_weights_to_keras_model(eval_model, state.model)

    ev_result = eval_model.evaluate(x_train, y_train, verbose=0)
    print('round {:2d}, metrics={}'.format(round_num, metrics))
    print(f"Eval loss : {ev_result[0]} and Eval accuracy : {ev_result[1]}")
    tff_train_acc.append(float(metrics.sparse_categorical_accuracy))
    tff_val_acc.append(ev_result[1])
    tff_train_loss.append(float(metrics.loss))
    tff_val_loss.append(ev_result[0])

eval_model.save(model_dir / "with_tff_model.h5")

#print(tff_train_acc, tff_val_acc, tff_train_loss, tff_val_loss)
fig = plt.figure(figsize=(10, 6))
plot_graph(list(range(1, 26))[4::5], tff_train_acc, label='Train Accuracy')
plot_graph(list(range(1, 26))[4::5], tff_val_acc, label='Validation Accuracy')
plt.legend()
plt.savefig(output_dir / "federated_model_Accuracy.png")

plt.figure(figsize=(10, 6))
plot_graph(list(range(1, 26))[4::5], tff_train_loss, label='Train loss')
plot_graph(list(range(1, 26))[4::5], tff_val_loss, label='Validation loss')
plt.legend()
plt.savefig(output_dir / "federated_model_loss.png")

plt.figure(figsize=(10, 6))
plot_graph(list(range(1, 26))[4::5], tff_train_acc, label='Federated Train Acc')
plot_graph(range(1, len(history.epoch)+1), history.history['sparse_categorical_accuracy'], label='Normal Train Acc')
plt.legend()
plt.savefig(output_dir / "federated_v_s_normal_model_train_Accuracy.png")

plt.figure(figsize=(10, 6))
plot_graph(list(range(1, 26))[4::5], tff_val_acc, label='Federated val Acc')
plot_graph(range(1, len(history.epoch)+1), history.history['val_sparse_categorical_accuracy'], label='Normal val Acc')
plt.legend()
plt.savefig(output_dir / "federated_v_s_normal_model_validation_Accuracy.png")
