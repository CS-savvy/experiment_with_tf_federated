# import collections
# import numpy as np
# import tensorflow as tf
# import tensorflow_federated as tff


#tf.compat.v1.enable_v2_behavior()
# np.random.seed(0)
# k = tff.federated_computation(lambda: 'Hello, World!')()
#
# emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
#
# example_dataset = emnist_train.create_tf_dataset_for_client(
#     emnist_train.client_ids[0])
#
# example_element = next(iter(example_dataset))
#
# print(example_element['label'].numpy())
# print(example_element['label'].numpy().shape)
# print(example_element['pixels'].numpy().shape)
#
# print()

# <TensorSliceDataset shapes: {pixels: (28, 28), labels: ()}, types: {pixels: tf.uint8, labels: tf.uint8}>
# {'pixels': TensorSpec(shape=(28, 28), dtype=tf.uint8, name=None), 'labels': TensorSpec(shape=(), dtype=tf.uint8, name=None)}

from matplotlib import pyplot as plt

def plot_graph(X, y, format = '-', label=''):
    plt.plot(X, y, format, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(True)


tff_train_acc = [0.691433310508728, 0.8867800235748291, 0.8963833451271057, 0.9032800197601318, 0.8948299884796143]
tff_val_acc = [0.2056, 0.2564, 0.331, 0.2146, 0.2041]
tff_train_loss = [0.8841511607170105, 0.42369043827056885, 0.37885618209838867, 0.3528357148170471, 0.3778982162475586]
tff_val_loss = [18.488481060791017, 3.1459058555603026, 2.1299105655670165, 2.2005297706604003, 2.195124142456055]


fig = plt.figure(figsize=(10, 6))
plot_graph(list(range(1, 26))[4::5], tff_train_acc, label='Train Accuracy')
plot_graph(list(range(1, 26))[4::5], tff_val_acc, label='Validation Accuracy')
plt.legend()
#plt.savefig("federated_model_Accuracy.png")
plt.show()

plt.figure(figsize=(10, 6))
plot_graph(list(range(1, 26))[4::5], tff_train_loss, label='Train loss')
plot_graph(list(range(1, 31))[4::5], tff_val_loss, label='Validation loss')
plt.legend()
#plt.savefig("federated_model_loss.png")
plt.show()