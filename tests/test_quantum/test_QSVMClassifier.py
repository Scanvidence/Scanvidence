import numpy as np
import pytest

from scanvidence.quantum import QSVMClassifier


def test_qsvm_fit_predict():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(20, 9))
    y = np.array([0] * 10 + [1] * 10)

    clf = QSVMClassifier(n_qubits=9, reps=1, C=1.0)
    clf.fit(X, y)

    preds = clf.predict(X[:5])
    probs = clf.predict_proba(X[:5])

    assert len(preds) == 5
    assert probs.shape == (5, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_qsvm_dimension_mismatch():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(10, 5))  # 5 features instead of 9
    y = np.array([0] * 5 + [1] * 5)

    clf = QSVMClassifier(n_qubits=9)
    with pytest.raises(ValueError, match="Feature dimension"):
        clf.fit(X, y)
