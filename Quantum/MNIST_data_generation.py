import tensorflow as tf
import collections
import numpy as np


def filter_03(x, y):
    keep = (y == 0) | (y == 3)
    x, y = x[keep], y[keep]
    y = y == 0
    return x,y



def truncate_x(x_train, x_test, n_components=10):
  """Perform PCA on image dataset keeping the top `n_components` components."""
  n_points_train = tf.gather(tf.shape(x_train), 0)
  n_points_test = tf.gather(tf.shape(x_test), 0)

  # Flatten to 1D
  x_train = tf.reshape(x_train, [n_points_train, -1])
  x_test = tf.reshape(x_test, [n_points_test, -1])

  # Normalize.
  feature_mean = tf.reduce_mean(x_train, axis=0)
  x_train_normalized = x_train - feature_mean
  x_test_normalized = x_test - feature_mean

  # Truncate.
  e_values, e_vectors = tf.linalg.eigh(
      tf.einsum('ji,jk->ik', x_train_normalized, x_train_normalized))
  return tf.einsum('ij,jk->ik', x_train_normalized, e_vectors[:,-n_components:]), \
    tf.einsum('ij,jk->ik', x_test_normalized, e_vectors[:, -n_components:])






def dataset(save_path):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Rescale the images from [0,255] to the [0.0,1.0] range.
    x_train, x_test = x_train/255.0, x_test/255.0

    x_train, y_train = filter_03(x_train, y_train)
    x_test, y_test = filter_03(x_test, y_test)

    DATASET_DIM = 10
    x_train, x_test = truncate_x(x_train, x_test, n_components=DATASET_DIM)

    N_TRAIN = 1000
    N_TEST = 200
    x_train, x_test = x_train[:N_TRAIN], x_test[:N_TEST]
    y_train, y_test = y_train[:N_TRAIN], y_test[:N_TEST]

    np.save(save_path + "/x_train.npy",x_train)
    np.save(save_path + "/x_test.npy",x_test)
    np.save(save_path + "/y_train.npy",y_train)
    np.save(save_path + "/y_test.npy",y_test)



dataset("/home/aleks/G2NK-Quantum-files/MNIST_Dataset")

    
