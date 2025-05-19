import os
import shutil
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random
from qiskit_machine_learning.algorithms import QSVC
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit_machine_learning.kernels import QuantumKernel
from sklearn.base import BaseEstimator

def custom_feature_map(num_qubits):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.h(i)
        qc.rz(0.5, i)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
        qc.rz(0.5, i + 1)
        qc.cx(i, i + 1)
    return qc

def load_data(L=500):
    known_dir = "normalized_lightcurves_known"
    unknown_dir = "normalized_lightcurves_unknown"
    def load_from_dir(directory, label):
        data = []
        filenames = []
        for file in os.listdir(directory):
            if file.endswith(".csv"):
                path = os.path.join(directory, file)
                try:
                    df = pd.read_csv(path)
                    if 'flux' not in df.columns:
                        continue
                    flux = df['flux'].values[:L]
                    if len(flux) < L:
                        flux = np.pad(flux, (0, L - len(flux)), 'constant', constant_values=0)
                    data.append((flux, label))
                    filenames.append(file)
                except:
                    continue
        return data, filenames
    
    # Load both known and unknown data separately
    known_data, known_files = load_from_dir(known_dir, 1)
    unknown_data, unknown_files = load_from_dir(unknown_dir, 0)

    return known_data, known_files, unknown_data, unknown_files 



def train_qsvc_classifier(known_data, known_files, unknown_data, unknown_files):
    # Shuffle and split known data
    np.random.seed(42)
    X_known, y_known = zip(*known_data)
    X_known = np.array(X_known)
    y_known = np.array(y_known)

    # Optionally scale
    X_known = StandardScaler().fit_transform(X_known)

    X_train, X_test, y_train, y_test = train_test_split(X_known, y_known, test_size=0.2, random_state=42)

    # Quantum kernel
    feature_map = custom_feature_map(feature_dimension=X_train.shape[1], reps=2)
    sampler = Sampler()
    kernel = QuantumKernel(feature_map=feature_map, sampler=sampler)

    # QSVC
    qsvc = QSVC(quantum_kernel=kernel)
    qsvc.fit(X_train, y_train)

    # Evaluate accuracy on known data
    accuracy = qsvc.score(X_test, y_test)
    print(f"Quantum classifier accuracy: {accuracy:.5f}")
    
    # Make predictions on the unknown data
    X_unknown = np.array([flux for flux, _ in unknown_data])  # Extract flux values for unknown
    X_unknown = StandardScaler().fit_transform(X_unknown)  # Optionally scale

    unknown_predictions = qsvc.predict(X_unknown)
    
    # Print out which unknown files are identified as possibly having transiting planets (label 1)
    for file, prediction in zip(unknown_files, unknown_predictions):
        os.makedirs("predicted_planets", exist_ok=True)
        if prediction == 1:
            print(f"{file} may have transit planet")
            source_path = os.path.join("normalized_lightcurves_unknown", file)  # Path to the original file
            destination_path = os.path.join("predicted_planets", file)  # New path in the predictions folder
            shutil.move(source_path, destination_path) 


if __name__ == "__main__":
    known_data, known_files, unknown_data, unknown_files = load_data()
    train_qsvc_classifier(known_data, known_files, unknown_data, unknown_files)
