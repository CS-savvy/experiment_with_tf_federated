## TensorFlow Federated Experiment

This repository is having python scripts to compare the effectiveness of the model trained with federated training methods with Non-federated training methods.

#### Training

1. To train model using TF-Federated, you can `train_tff.py`. You can find the hyperparameters in first few line of code.
2. use `train_keras.py` to start training with usual keras non-federated training.
3. you can also use `train_combined.py` to train one by one both methods with same dataset and model.

#### Compare

Run `compare_results.py` to generate comparison graph for you metrics as it is saved in `results` folder in there respective sub-folders.

##### sample plots

Loss                       |  Accuracy
:-------------------------:|:-------------------------:
![loss](https://github.com/CS-savvy/experiment_with_tf_federated/blob/master/results/mnist/compare/loss.png?raw=true) | ![loss](https://github.com/CS-savvy/experiment_with_tf_federated/blob/master/results/mnist/compare/sparse_categorical_accuracy.png?raw=true)
![loss](https://github.com/CS-savvy/experiment_with_tf_federated/blob/master/results/mnist/compare/val_loss.png?raw=true) |  ![loss](https://github.com/CS-savvy/experiment_with_tf_federated/blob/master/results/mnist/compare/val_sparse_categorical_accuracy.png?raw=true)
