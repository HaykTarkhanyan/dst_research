import numpy as np
import pytest

from src.DSClassifierMultiQ import DSClassifierMultiQ

SEED = 509
N_SAMPLES = 8  # per class
MAX_ITER = 3   # keep training minimal


class TestClassifierInit:
    def test_default_attributes(self):
        clf = DSClassifierMultiQ(num_classes=2)
        assert clf.k == 2
        assert clf.model.k == 2
        assert clf.classes_ == [0, 1]

    def test_multiclass(self):
        clf = DSClassifierMultiQ(num_classes=5)
        assert clf.k == 5
        assert clf.classes_ == [0, 1, 2, 3, 4]


@pytest.fixture(scope="module")
def fitted_binary_clf():
    """Train a binary classifier once, reuse across all tests."""
    np.random.seed(SEED)
    rng = np.random.RandomState(SEED)
    n = N_SAMPLES
    X = np.vstack([rng.randn(n, 2) + [-2, -2], rng.randn(n, 2) + [2, 2]])
    y = np.array([0] * n + [1] * n)
    clf = DSClassifierMultiQ(num_classes=2, lr=0.01, max_iter=MAX_ITER, min_dloss=-1, lossfn="MSE")
    losses, epoch = clf.fit(X, y, add_single_rules=True, single_rules_breaks=2)
    return clf, X, y, losses, epoch


@pytest.fixture(scope="module")
def fitted_binary_clf_ce():
    """Train binary classifier with CE + mult rules once."""
    np.random.seed(SEED)
    rng = np.random.RandomState(SEED)
    n = N_SAMPLES
    X = np.vstack([rng.randn(n, 2) + [-2, -2], rng.randn(n, 2) + [2, 2]])
    y = np.array([0] * n + [1] * n)
    clf = DSClassifierMultiQ(num_classes=2, lr=0.01, max_iter=MAX_ITER, lossfn="CE")
    losses, epoch = clf.fit(X, y, add_single_rules=True, single_rules_breaks=2, add_mult_rules=True)
    return clf, X, y, losses, epoch


@pytest.fixture(scope="module")
def fitted_multiclass_clf():
    """Train a 3-class classifier once."""
    np.random.seed(SEED)
    rng = np.random.RandomState(SEED)
    n = 5
    X = np.vstack([rng.randn(n, 2) + [-3, 0], rng.randn(n, 2) + [3, 0], rng.randn(n, 2) + [0, 3]])
    y = np.array([0] * n + [1] * n + [2] * n)
    clf = DSClassifierMultiQ(num_classes=3, lr=0.01, max_iter=MAX_ITER, lossfn="CE")
    clf.fit(X, y, add_single_rules=True, single_rules_breaks=2)
    return clf, X, y


class TestClassifierFitPredict:
    def test_fit_returns_losses(self, fitted_binary_clf):
        _, _, _, losses, epoch = fitted_binary_clf
        assert len(losses) > 0
        assert isinstance(epoch, int)

    def test_predict_shape(self, fitted_binary_clf):
        clf, X, _, _, _ = fitted_binary_clf
        assert clf.predict(X).shape == (len(X),)

    def test_predict_valid_classes(self, fitted_binary_clf):
        clf, X, _, _, _ = fitted_binary_clf
        assert set(clf.predict(X)).issubset({0, 1})

    def test_predict_proba_shape(self, fitted_binary_clf):
        clf, X, _, _, _ = fitted_binary_clf
        assert clf.predict_proba(X).shape == (len(X), 2)

    def test_predict_proba_sums_to_one(self, fitted_binary_clf):
        clf, X, _, _, _ = fitted_binary_clf
        proba = clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-4)

    def test_cross_entropy_loss(self, fitted_binary_clf_ce):
        _, _, _, losses, _ = fitted_binary_clf_ce
        assert len(losses) > 0


class TestClassifierMulticlass:
    def test_multiclass_fit_predict(self, fitted_multiclass_clf):
        clf, X, _ = fitted_multiclass_clf
        preds = clf.predict(X)
        assert preds.shape == (len(X),)
        assert set(preds).issubset({0, 1, 2})

    def test_multiclass_proba_shape(self, fitted_multiclass_clf):
        clf, X, _ = fitted_multiclass_clf
        assert clf.predict_proba(X).shape == (len(X), 3)


class TestPredictExplain:
    def test_explain_returns_correct_structure(self, fitted_binary_clf):
        clf, X, _, _, _ = fitted_binary_clf
        pred, cls, df_rls, builder = clf.predict_explain(X[0])
        assert len(pred) == 2
        assert cls in [0, 1]
        assert "rule" in df_rls.columns
        assert "uncertainty" in df_rls.columns
        assert isinstance(builder, str)


class TestOptimizerOptions:
    def test_sgd_optimizer(self, binary_dataset):
        X, y = binary_dataset
        clf = DSClassifierMultiQ(num_classes=2, lr=0.01, max_iter=2, optim="sgd", lossfn="MSE")
        losses, _ = clf.fit(X, y, add_single_rules=True, single_rules_breaks=2)
        assert len(losses) > 0

    def test_invalid_optimizer_raises(self, binary_dataset):
        X, y = binary_dataset
        clf = DSClassifierMultiQ(num_classes=2, optim="invalid")
        with pytest.raises(RuntimeError, match="Unknown optimizer"):
            clf.fit(X, y, add_single_rules=True, single_rules_breaks=2)

    def test_invalid_lossfn_raises(self, binary_dataset):
        X, y = binary_dataset
        clf = DSClassifierMultiQ(num_classes=2, lossfn="invalid")
        with pytest.raises(RuntimeError, match="Unknown loss function"):
            clf.fit(X, y, add_single_rules=True, single_rules_breaks=2)
