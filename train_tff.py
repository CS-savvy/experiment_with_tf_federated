import collections
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow import reshape, nest, config
from tensorflow.keras import losses, metrics, optimizers
import tensorflow_federated as tff
from matplotlib import pyplot as plt
from models.conv_2 import create_keras_model
from pathlib import Path
from utils import plot_graph

gpus = config.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
        config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


experiment_name = "mnist"
method = "tff_training"
model_name = "conv_simple_tff.h5"
client_lr = 1e-2
server_lr = 1e-2
split = 4
NUM_ROUNDS = 5
NUM_EPOCHS = 5
BATCH_SIZE = 20
PREFETCH_BUFFER = 10

this_dir = Path.cwd()
model_dir = this_dir / "saved_models" / experiment_name / method
output_dir = this_dir / "results" / experiment_name / method

if not model_dir.exists():
    model_dir.mkdir(parents=True)

if not output_dir.exists():
    output_dir.mkdir(parents=True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.int32)
x_test = x_test.astype(np.float32).reshape(10000, 28, 28, 1)
y_test = y_test.astype(np.int32).reshape(10000, 1)

total_image_count = len(x_train)
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

sample_dataset = train_dataset.create_tf_dataset_for_client(train_dataset.client_ids[0])
sample_element = next(iter(sample_dataset))

SHUFFLE_BUFFER = image_per_set

def preprocess(dataset):

  def batch_format_fn(element):
    """Flatten a batch `pixels` and return the features as an `OrderedDict`."""

    return collections.OrderedDict(
        x=reshape(element['pixels'], [-1, 28, 28, 1]),
        y=reshape(element['label'], [-1, 1]))

  return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(
      BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)

preprocessed_sample_dataset = preprocess(sample_dataset)
sample_batch = nest.map_structure(lambda x: x.numpy(), next(iter(preprocessed_sample_dataset)))


def make_federated_data(client_data, client_ids):
    return [preprocess(client_data.create_tf_dataset_for_client(x)) for x in client_ids]

federated_train_data = make_federated_data(train_dataset, train_dataset.client_ids)

print('Number of client datasets: {l}'.format(l=len(federated_train_data)))
print('First dataset: {d}'.format(d=federated_train_data[0]))

def model_fn():
  # We _must_ create a new model here, and _not_ capture it from an external
  # scope. TFF will call this within different graph contexts.

  keras_model = create_keras_model()
  return tff.learning.from_keras_model(
      keras_model,
      input_spec=preprocessed_sample_dataset.element_spec,
      loss=losses.SparseCategoricalCrossentropy(),
      metrics=[metrics.SparseCategoricalAccuracy()])


iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: optimizers.Adam(learning_rate=client_lr),
    server_optimizer_fn=lambda: optimizers.SGD(learning_rate=server_lr))

print(str(iterative_process.initialize.type_signature))

state = iterative_process.initialize()

tff_train_acc = []
tff_val_acc = []
tff_train_loss = []
tff_val_loss = []

eval_model = None
for round_num in range(1, NUM_ROUNDS+1):
    state, tff_metrics = iterative_process.next(state, federated_train_data)
    eval_model = create_keras_model()
    eval_model.compile(optimizer=optimizers.Adam(learning_rate=client_lr),
                       loss=losses.SparseCategoricalCrossentropy(),
                       metrics=[metrics.SparseCategoricalAccuracy()])

    tff.learning.assign_weights_to_keras_model(eval_model, state.model)

    ev_result = eval_model.evaluate(x_test, y_test, verbose=0)
    print('round {:2d}, metrics={}'.format(round_num, tff_metrics))
    print(f"Eval loss : {ev_result[0]} and Eval accuracy : {ev_result[1]}")
    tff_train_acc.append(float(tff_metrics.sparse_categorical_accuracy))
    tff_val_acc.append(ev_result[1])
    tff_train_loss.append(float(tff_metrics.loss))
    tff_val_loss.append(ev_result[0])

if eval_model:
    eval_model.save(model_dir / model_name)
else:
    print("training didn't started")
    exit()

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



