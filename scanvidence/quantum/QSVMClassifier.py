"""Quantum Kernel Support Vector Machine (QSVM) for molecular biomarker prediction.

Embeds radiomic feature vectors into a 2^n-dimensional complex Hilbert space
using parameterized quantum feature maps (ZZFeatureMap) and trains a dual-space
support vector classifier with calibrated probability outputs.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from qiskit.circuit.library import PauliFeatureMap, ZZFeatureMap
from qiskit.quantum_info import Statevector
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


class QSVMClassifier(BaseEstimator, ClassifierMixin):
    """Quantum Kernel Support Vector Classifier (QSVM).

    Parameters
    ----------
    n_qubits : int, default=9
        Number of qubits (feature dimension).
    feature_map_type : {'ZZFeatureMap', 'PauliFeatureMap'}, default='ZZFeatureMap'
        Quantum circuit feature mapping.
    reps : int, default=1
        Number of feature map layer repetitions.
    entanglement : {'linear', 'circular', 'full'}, default='linear'
        Entanglement topology for two-qubit interactions.
    C : float, default=1.0
        Regularization parameter for SVM.
    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_qubits: int = 9,
        feature_map_type: Literal["ZZFeatureMap", "PauliFeatureMap"] = "ZZFeatureMap",
        reps: int = 1,
        entanglement: Literal["linear", "circular", "full"] = "linear",
        C: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.n_qubits = n_qubits
        self.feature_map_type = feature_map_type
        self.reps = reps
        self.entanglement = entanglement
        self.C = C
        self.random_state = random_state

        self.scaler = MinMaxScaler(feature_range=(0, 2 * np.pi))
        self.feature_map = self._build_feature_map()
        self.svc = SVC(
            kernel="precomputed",
            C=self.C,
            probability=True,
            random_state=self.random_state,
        )
        self.train_states_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    def _build_feature_map(self):
        if self.feature_map_type == "ZZFeatureMap":
            return ZZFeatureMap(
                feature_dimension=self.n_qubits,
                reps=self.reps,
                entanglement=self.entanglement,
            )
        elif self.feature_map_type == "PauliFeatureMap":
            return PauliFeatureMap(
                feature_dimension=self.n_qubits,
                reps=self.reps,
                entanglement=self.entanglement,
            )
        else:
            raise ValueError(f"Unknown feature map type: {self.feature_map_type}")

    def _compute_quantum_states(self, X_scaled: np.ndarray) -> np.ndarray:
        """Map scaled classical vectors to 2^n statevectors."""
        states = []
        for row in X_scaled:
            qc = self.feature_map.assign_parameters(row)
            sv = Statevector(qc)
            states.append(sv.data)
        return np.array(states)

    def _compute_kernel_matrix(self, states_a: np.ndarray, states_b: np.ndarray) -> np.ndarray:
        """Compute Quantum Kernel Fidelity matrix: K_ij = |<Phi(x_i)|Phi(x_j)>|^2."""
        overlap = np.dot(states_a, states_b.conj().T)
        return np.abs(overlap) ** 2

    def fit(self, X: np.ndarray, y: np.ndarray) -> QSVMClassifier:
        """Fit Quantum Kernel SVM model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target class labels.
        """
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y_arr)

        if X_arr.shape[1] != self.n_qubits:
            raise ValueError(
                f"Feature dimension {X_arr.shape[1]} does not match n_qubits {self.n_qubits}"
            )

        X_scaled = self.scaler.fit_transform(X_arr)
        self.train_states_ = self._compute_quantum_states(X_scaled)

        K_train = self._compute_kernel_matrix(self.train_states_, self.train_states_)
        self.svc.fit(K_train, y_arr)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X using the Quantum Kernel."""
        if self.train_states_ is None:
            raise ValueError("Model is not fitted yet.")

        X_arr = np.asarray(X, dtype=float)
        X_scaled = self.scaler.transform(X_arr)
        test_states = self._compute_quantum_states(X_scaled)

        K_test = self._compute_kernel_matrix(test_states, self.train_states_)
        return self.svc.predict_proba(K_test)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X."""
        if self.classes_ is None:
            raise ValueError("Model is not fitted yet.")
        probs = self.predict_proba(X)
        return np.asarray(self.classes_)[np.argmax(probs, axis=1)]

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the decision function for X."""
        if self.train_states_ is None:
            raise ValueError("Model is not fitted yet.")

        X_arr = np.asarray(X, dtype=float)
        X_scaled = self.scaler.transform(X_arr)
        test_states = self._compute_quantum_states(X_scaled)

        K_test = self._compute_kernel_matrix(test_states, self.train_states_)
        return self.svc.decision_function(K_test)
