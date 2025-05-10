import jax
import pennylane as qml
import numpy as np
import numpy as np

np.random.seed(1234)

def extract_pkq_features(x):
    X = x
    N = 11
    trotter_number = 10
    # create device using JAX
    device = qml.device("default.qubit", wires=N)
    @jax.jit
    @qml.qnode(device, interface="jax")
    
    def get_features(x):
        wires = range(N)
        qml.AngleEmbedding(x, wires=wires, rotation="Y")
        # random rotations
        for _ in range(trotter_number):
            for i in range(len(wires) - 1):
                angle = np.random.normal()
                qml.IsingXX(angle, wires=[wires[i], wires[i + 1]])
                qml.IsingYY(angle, wires=[wires[i], wires[i + 1]])
                qml.IsingZZ(angle, wires=[wires[i], wires[i + 1]])

        return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0)), \
                qml.expval(qml.PauliX(1)), qml.expval(qml.PauliY(1)), qml.expval(qml.PauliZ(1)), \
                qml.expval(qml.PauliX(2)), qml.expval(qml.PauliY(2)), qml.expval(qml.PauliZ(2)), \
                qml.expval(qml.PauliX(3)), qml.expval(qml.PauliY(3)), qml.expval(qml.PauliZ(3)), \
                qml.expval(qml.PauliX(4)), qml.expval(qml.PauliY(4)), qml.expval(qml.PauliZ(4)), \
                qml.expval(qml.PauliX(5)), qml.expval(qml.PauliY(5)), qml.expval(qml.PauliZ(5)), \
                qml.expval(qml.PauliX(6)), qml.expval(qml.PauliY(6)), qml.expval(qml.PauliZ(6)), \
                qml.expval(qml.PauliX(7)), qml.expval(qml.PauliY(7)), qml.expval(qml.PauliZ(7)), \
                qml.expval(qml.PauliX(8)), qml.expval(qml.PauliY(8)), qml.expval(qml.PauliZ(8)), \
                qml.expval(qml.PauliX(9)), qml.expval(qml.PauliY(9)), qml.expval(qml.PauliZ(9)), \
                qml.expval(qml.PauliX(10)), qml.expval(qml.PauliY(10)), qml.expval(qml.PauliZ(10))

    x_pqk = np.array([get_features(x) for x in X])
    #print(np.shape(x_pqk))
    a1,a2,a3 = np.array_split(x_pqk,3,axis=1)
    rdm = np.stack((a1,a2,a3),axis=2)
    #print(np.shape(rdm))
    return rdm,x_pqk