from tkinter import N
from turtle import xcor
import jax
import jax.numpy as jnp
import pennylane as qml
import numpy as np
import os
from sklearn.metrics.pairwise import rbf_kernel
from sklearn import preprocessing
from sklearn import decomposition

import numpy as np
from sklearn.svm import SVC
from scipy.linalg import sqrtm
import numpy.linalg as la
from PQK_features_extraction import *

np.random.seed(1234)

def compute_gram_matrix(data, gamma=1.0):
    gram = np.zeros(shape=(data.shape[0], data.shape[0]))
    # build the gram matrix
    scaled_gamma = gamma / (
        np.float32(data.shape[1]) * np.std(data)
    )
    #print(scaled_gamma)
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            value = np.exp(-scaled_gamma * ((data[i] - data[j]) ** 2).sum())
            gram[i][j] = value
    return gram
    
def stilted_dataset(gram,base_gram,lambdav=1.1):
    S,V = la.eigh(gram)
    #print(S)
    S_CLASSIC, V_CLASSIC = la.eigh(base_gram)
    S_CLASSIC = np.abs(S_CLASSIC)
    S_diag = np.diag(S**0.5)
    #print(np.shape(S_diag))           
    #print(np.sum(~np.isnan(S_diag)))
    #print(S_diag)
    S_CLASSIC_diag = np.diag(S_CLASSIC/(S_CLASSIC+lambdav) ** 2)
    scaling = S_diag @ np.transpose(V) @ \
        V_CLASSIC @ S_CLASSIC_diag @ np.transpose(V_CLASSIC) @\
        V @ S_diag
    _,vectors = la.eig(scaling)
    #print(type(vectors))
    new_labels = np.real(
        np.einsum('ij,j->i', (V @ S_diag).astype(np.complex64), vectors[-1])
    )
    final_y = new_labels > np.median(new_labels)
    noisy_y =((final_y ^ (np.random.uniform(size=final_y.shape) > 0.95)))
    #print(np.shape(S),np.shape(V),np.shape(S_CLASSIC),np.shape(V_CLASSIC),print(noisy_y))
    return noisy_y

def GW_data_loading(path_sig,path_labels,nbr_train):
    
    # Loading
    signal = np.load(path_sig)
    labels = np.load(path_labels)

    # PCA
    pca = decomposition.PCA(n_components=10)
    signal_pca = pca.fit_transform(signal)

    # Scaling
    min_max_scaler = preprocessing.MinMaxScaler()
    signal_scaled = min_max_scaler.fit_transform(signal_pca)
    signal_final = preprocessing.normalize(signal_scaled)
    
    x_train = signal_final[:nbr_train,:]
    y_train = labels[:nbr_train]
    x_test = signal_final[nbr_train:,:]
    y_test  = labels[nbr_train:]
    return x_train,y_train,x_test,y_test


def experiment_MNIST(path):
    x_train = np.load(path+"/Notebook_dataset/x_train.npy")
    x_test = np.load(path+"/Notebook_dataset/x_test.npy")
    x_train_pqk, _ = extract_pkq_features(x_train)
    x_test_pqk, _ = extract_pkq_features(x_test)
    base_gram = compute_gram_matrix(np.concatenate([x_train,x_test],0))
    gram = compute_gram_matrix(np.concatenate([x_train_pqk,x_test_pqk],0))
    y_relabel = stilted_dataset(gram,base_gram)
    y_train_new_MNIST, y_test_new_MNIST = y_relabel[:N_TRAIN], y_relabel[N_TRAIN:]
    return x_train_pqk,x_test_pqk,y_train_new_MNIST,y_test_new_MNIST


def experiment_GW(path_sig, path_labels):
    x_train,y_train,x_test,y_test = GW_data_loading(path_sig,path_labels,N_TRAIN)
    _, x_train_pqk = extract_pkq_features(x_train)
    _, x_test_pqk = extract_pkq_features(x_test)
    base_gram = compute_gram_matrix(np.concatenate([x_train,x_test],0))
    gram = compute_gram_matrix(np.concatenate([x_train_pqk,x_test_pqk],0))
    y_relabel = stilted_dataset(gram,base_gram)
    y_train_new_GW, y_test_new_GW = y_relabel[:N_TRAIN], y_relabel[N_TRAIN:]
    return x_train_pqk,x_test_pqk,y_train_new_GW,y_test_new_GW,y_train,y_test



if __name__ ==  "__main__":
    N_TRAIN = 800 
    N_TEST = 200
    path = os.getcwd()+"/Quantum"
    x_train_pqk_MNIST,x_test_pqk_MNIST,y_train_new_MNIST,y_test_new_MNIST = experiment_MNIST(path)
    np.save("./Data/x_train_pqk_pennylane.npy",x_train_pqk_MNIST)
    np.save("./Data/x_test_pqk_pennylane.npy", x_test_pqk_MNIST)
    np.save("./Data/y_train_new_pennylane_MNIST.npy",y_train_new_MNIST)
    np.save("./Data/y_test_new_MNIST.npy", y_test_new_MNIST)

    #path_sig = "/media/aleks/USB STICK/extracted_features/all_files.npy"
    #path_labels = "/media/aleks/USB STICK/labels.npy"
    #print(y_train_new)
    print(np.shape(x_train_pqk_MNIST))
