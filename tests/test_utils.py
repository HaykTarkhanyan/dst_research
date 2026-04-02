import numpy as np
import pytest

from src.utils import (
    calculate_adjusted_density,
    detect_outliers_z_score,
    evaluate_classifier,
    is_categorical,
    normalize,
    one_hot,
    h_center,
    h_left,
    h_right,
)


class TestIsCategorical:
    def test_categorical_array(self):
        assert is_categorical(np.array([1, 2, 3, 1, 2, 3])) is True

    def test_continuous_array(self):
        rng = np.random.RandomState(509)
        assert is_categorical(rng.randn(50)) is False

    def test_with_nans(self):
        assert is_categorical(np.array([1.0, 2.0, np.nan, 1.0, 2.0])) is True

    def test_custom_threshold(self):
        arr = np.array([1, 2, 3, 4, 5, 6, 7])
        assert is_categorical(arr, max_cat=6) is False
        assert is_categorical(arr, max_cat=10) is True


class TestNormalize:
    def test_sums_to_one(self):
        a, b, c = normalize(0.3, 0.4, 0.3)
        assert pytest.approx(a + b + c, abs=1e-9) == 1.0

    def test_clamps_negatives(self):
        a, b, c = normalize(-0.5, 0.8, 0.3)
        assert a >= 0

    def test_clamps_above_one(self):
        a, b, c = normalize(1.5, 0.3, 0.2)
        assert a <= 1.0 / (1.0 + 0.3 + 0.2) + 1e-9


class TestOneHot:
    def test_shape(self):
        assert one_hot(np.array([0, 1, 2, 0]), 3).shape == (4, 3)

    def test_values(self):
        result = one_hot(np.array([0, 1]), 2)
        np.testing.assert_array_equal(result[0], [1, 0])
        np.testing.assert_array_equal(result[1], [0, 1])

    def test_row_sums(self):
        result = one_hot(np.array([0, 1, 2]), 3)
        np.testing.assert_array_equal(result.sum(axis=1), [1, 1, 1])


class TestDetectOutliersZScore:
    def test_no_outliers_in_tight_data(self):
        assert detect_outliers_z_score(np.ones(20)) == []

    def test_detects_obvious_outlier(self):
        data = np.concatenate([np.zeros(20), [100.0]])
        assert 100.0 in detect_outliers_z_score(data, threshold=2)

    def test_threshold_sensitivity(self):
        data = np.array([0, 0, 0, 0, 0, 10])
        loose = detect_outliers_z_score(data, threshold=3)
        tight = detect_outliers_z_score(data, threshold=1)
        assert len(tight) >= len(loose)


class TestCalculateAdjustedDensity:
    def test_output_length(self):
        data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        labels = np.array([0, 0, 1, 1])
        assert len(calculate_adjusted_density(data, labels, radius=1.5)) == 4

    def test_same_class_neighbors_increase_density(self):
        data = np.array([[0, 0], [0.1, 0], [0, 0.1], [10, 10]])
        labels = np.array([0, 0, 0, 1])
        densities = calculate_adjusted_density(data, labels, radius=1.0)
        assert densities[0] > densities[3]

    def test_raises_on_zero_density(self):
        data = np.array([[0, 0], [100, 100]])
        labels = np.array([0, 1])
        with pytest.raises(ValueError, match="All densities are zero"):
            calculate_adjusted_density(data, labels, radius=0.001)


class TestEvaluateClassifier:
    def test_perfect_predictions(self):
        y = np.array([0, 0, 1, 1])
        res = evaluate_classifier(y_actual=y, y_clust=y)
        assert res["accuracy"] == 1.0
        assert res["f1"] == 1.0

    def test_swapped_labels_corrected(self):
        y_actual = np.array([0, 0, 1, 1])
        y_clust = np.array([1, 1, 0, 0])
        res = evaluate_classifier(y_actual=y_actual, y_clust=y_clust, purpose="kmeans_eval")
        assert res["accuracy"] == 1.0

    def test_accuracy_range(self):
        rng = np.random.RandomState(509)
        y = rng.randint(0, 2, 30)
        y_pred = rng.randint(0, 2, 30)
        res = evaluate_classifier(y_actual=y, y_clust=y_pred, purpose="eval")
        assert 0.0 <= res["accuracy"] <= 1.0


class TestHelperFunctions:
    def test_h_center_at_zero(self):
        assert h_center(0) == 1.0

    def test_h_center_decays(self):
        assert h_center(2) < h_center(0)

    def test_h_right_monotonic(self):
        vals = [h_right(z) for z in np.linspace(-3, 3, 10)]
        assert all(vals[i] <= vals[i + 1] + 1e-10 for i in range(len(vals) - 1))

    def test_h_left_monotonic(self):
        vals = [h_left(z) for z in np.linspace(-3, 3, 10)]
        assert all(vals[i] >= vals[i + 1] - 1e-10 for i in range(len(vals) - 1))
