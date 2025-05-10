from tkinter import N
from turtle import xcor
import jax
import jax.numpy as jnp
import pennylane as qml
import numpy as np
import optax
from sklearn.metrics.pairwise import rbf_kernel
from sklearn import preprocessing

import numpy as np
from sklearn.svm import SVC
from scipy.linalg import sqrtm
import numpy.linalg as la


def ry_embedding(x, wires):
    """
    Encode the data with one rotation on sigma_y per qubit per feature
    Args:
        x: feature vector (list or numpy array of floats)
        wires: wires of the circuit (list of int)
    Returns:
        None
    """
    qml.AngleEmbedding(x, wires=wires, rotation="Y")


def random_qnn_encoding(x, wires, trotter_number=10):
    """
    This function creates and appends a quantum neural network to the selected
    encoding. It follows formula S(116) in the Supplementary.
    Args:
        x: feature vector (list or numpy array of floats)
        wires: wires of the circuit (list of int)
        trotter_number: number of repetitions (int)
    Returns:
        None
    """
    assert len(x) == len(wires)
    # embedding
    ry_embedding(x, wires)
    # random rotations
    for _ in range(trotter_number):
        for i in range(len(wires) - 1):
            angle = np.random.normal()
            qml.IsingXX(angle, wires=[wires[i], wires[i + 1]])
            qml.IsingYY(angle, wires=[wires[i], wires[i + 1]])
            qml.IsingZZ(angle, wires=[wires[i], wires[i + 1]])


def projected_xyz_embedding(embedding_template, X):
    """
    Create a Quantum Kernel given the template written in Pennylane framework
    Args:
        embedding: Pennylane template for the quantum feature map
        X: feature data (matrix)
    Returns:
        projected quantum feature map X
    """
    N = X.shape[1]

    # create device using JAX
    device = qml.device("default.qubit", wires=N)

    # define the circuit for the quantum kernel ("overlap test" circuit)
    @jax.jit
    @qml.qnode(device, interface="jax")
    
    def proj_feature_map(x):
        embedding_template(x, wires=range(N))
        return (
            [qml.expval(qml.PauliX(i)) for i in range(N)]
            + [qml.expval(qml.PauliY(i)) for i in range(N)]
            + [qml.expval(qml.PauliZ(i)) for i in range(N)]
        )

    X_proj = [proj_feature_map(x) for x in X]

    return np.array(X_proj)


def pennylane_projected_quantum_kernel(embedding, X_1, X_2=None, wires=None , params=[1.0]):
    """
    Create a Quantum Kernel given the template written in Pennylane framework.
    Args:
        feature_map: Pennylane template for the quantum feature map
        X_1: First dataset
        X_2: Second dataset
        params: List of one single parameter representing the constant in the exponentiation
    Returns:
        Gram matrix
    """
    if X_2 is None:
        X_2 = X_1  # Training Gram matrix
    if wires is None:
        wires = X_1.shape[1]
    assert (
        X_1.shape[1] == X_2.shape[1]
    ), "The training and testing data must have the same dimensionality"

    X_1_proj = projected_xyz_embedding(embedding, X_1)
    X_2_proj = projected_xyz_embedding(embedding, X_2)

    # build the gram matrix
    gamma = params[0]

    gram = np.zeros(shape=(X_1.shape[0], X_2.shape[0]))
    for i in range(X_1_proj.shape[0]):
        for j in range(X_2_proj.shape[0]):
            value = np.exp(-gamma * ((X_1_proj[i] - X_2_proj[j]) ** 2).sum())
            gram[i][j] = value

    return gram



def calculate_geometric_difference(k_1, k_2, normalization_lambda=0.001):
    """
    Calculate the geometric difference g(K_1 || K_2), which is equation F9 in
    "The power of data in quantum machine learning" (https://arxiv.org/abs/2011.01938)
    and characterize the separation between classical and quantum kernels.

    Args:
        k_1: Quantum kernel Gram matrix
        k_2: Classical kernel Gram matrix
        normalization_lambda: normalization factor, must be close to zero

    Returns:
        geometric difference between the two kernel functions (float).
    """
    n = k_2.shape[0]
    assert k_2.shape == (n, n)
    assert k_1.shape == (n, n)
    # √K1
    k_1_sqrt = np.real(sqrtm(k_1))
    # √K2
    k_2_sqrt = np.real(sqrtm(k_2))
    # √(K2 + lambda I)^-2
    kc_inv = la.inv(k_2 + normalization_lambda * np.eye(n))
    kc_inv = kc_inv @ kc_inv
    # Equation F9
    f9_body = k_1_sqrt.dot(k_2_sqrt.dot(kc_inv.dot(k_2_sqrt.dot(k_1_sqrt))))
    f9 = np.sqrt(la.norm(f9_body, np.inf))
    return f9


def calculate_generalization_accuracy(
    training_gram, training_labels, testing_gram, testing_labels
):
    """
    Calculate accuracy wrt a precomputed kernel, a training and testing set

    Args:
        training_gram: Gram matrix of the training set, must have shape (N,N)
        training_labels: Labels of the training set, must have shape (N,)
        testing_gram: Gram matrix of the testing set, must have shape (M,N)
        testing_labels: Labels of the training set, must have shape (M,)

    Returns:
        generalization accuracy (float)
    """
    svm = SVC(kernel="precomputed")
    svm.fit(training_gram, training_labels)
    y_predict = svm.predict(testing_gram)
    correct = np.sum(testing_labels == y_predict)
    accuracy = correct / len(testing_labels)
    return accuracy


def experiment(x_train):
    base_gram_matrix = rbf_kernel(x_train)
    embedding = random_qnn_encoding
    gram_train = pennylane_projected_quantum_kernel(embedding=embedding,X_1=x_train)
    print(gram_train)
    #print(base_gram_matrix)
    np.save("./gram_train.npy",gram_train)
    geometric_difference = calculate_geometric_difference(gram_train,base_gram_matrix)
    print(geometric_difference)

def data_loading(path_sig,path_labels,portion_samples):
    
    # Loading
    signal = np.load(path_sig)
    labels = np.load(path_labels)

        # Scaling
    min_max_scaler = preprocessing.MinMaxScaler()
    signal_scaled = min_max_scaler.fit_transform(signal)

    nbr_train = int((1-portion_samples) * np.shape(signal)[0])

    x_train = signal_scaled[:nbr_train,:]
    y_train = labels[:nbr_train]
    x_test = signal_scaled[nbr_train:,:]
    y_test  = labels[nbr_train:]

    return x_train,y_train,x_test,y_test

if __name__ ==  "__main__":
    path = "/home/aleks/G2NK-Quantum-files/test_dataset/Output"
    x_train = np.load(path + "/X_train.npy")
    #x_train,y_train,x_test,y_test  = data_loading("/media/aleks/USB STICK/extracted_features/all_red.npy","/media/aleks/USB STICK/labels.npy",0.2)
    gram = pennylane_projected_quantum_kernel(random_qnn_encoding,x_train,wires=5)
    print(gram)
