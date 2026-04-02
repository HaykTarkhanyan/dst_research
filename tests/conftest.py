import numpy as np
import pytest

SEED = 509


@pytest.fixture
def binary_dataset():
    """Minimal linearly separable 2-class dataset."""
    rng = np.random.RandomState(SEED)
    n = 8
    X_class0 = rng.randn(n, 2) + np.array([-2, -2])
    X_class1 = rng.randn(n, 2) + np.array([2, 2])
    X = np.vstack([X_class0, X_class1])
    y = np.array([0] * n + [1] * n)
    return X, y


@pytest.fixture
def multiclass_dataset():
    """Minimal 3-class dataset."""
    rng = np.random.RandomState(SEED)
    n = 5
    X0 = rng.randn(n, 2) + np.array([-3, 0])
    X1 = rng.randn(n, 2) + np.array([3, 0])
    X2 = rng.randn(n, 2) + np.array([0, 3])
    X = np.vstack([X0, X1, X2])
    y = np.array([0] * n + [1] * n + [2] * n)
    return X, y
