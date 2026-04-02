"""Tests for uncertainty loop, rule uncertainty scores, ensemble, and pruning."""
import numpy as np
import pytest
import torch

from src.DSClassifierMultiQ import DSClassifierMultiQ
from src.DSModelMultiQ import DSModelMultiQ
from src.DSRule import DSRule
from src.rule_miners import DecisionTreeMiner
from src.uncertainty_loop import UncertaintyGuidedRefiner

SEED = 509
MAX_ITER = 3


@pytest.fixture
def binary_dataset():
    rng = np.random.RandomState(SEED)
    n = 30
    X_class0 = rng.randn(n, 2) + np.array([-2, -2])
    X_class1 = rng.randn(n, 2) + np.array([2, 2])
    X = np.vstack([X_class0, X_class1])
    y = np.array([0] * n + [1] * n)
    return X, y


# ---------- get_rule_uncertainty_scores ----------

class TestGetRuleUncertaintyScores:
    def test_returns_list_of_dicts(self, binary_dataset):
        X, y = binary_dataset
        clf = DSClassifierMultiQ(num_classes=2, max_iter=MAX_ITER, lossfn="CE")
        clf.fit(X, y, add_single_rules=True, single_rules_breaks=2)
        scores = clf.model.get_rule_uncertainty_scores()

        assert isinstance(scores, list)
        assert len(scores) == clf.model.get_rules_size()
        for s in scores:
            assert "rule_idx" in s
            assert "pred" in s
            assert "caption" in s
            assert "uncertainty_mass" in s
            assert "max_class_mass" in s
            assert "confidence_score" in s
            assert "masses" in s

    def test_confidence_score_formula(self, binary_dataset):
        X, y = binary_dataset
        clf = DSClassifierMultiQ(num_classes=2, max_iter=MAX_ITER, lossfn="CE")
        clf.fit(X, y, add_single_rules=True, single_rules_breaks=2)
        scores = clf.model.get_rule_uncertainty_scores()

        for s in scores:
            expected = s["max_class_mass"] * (1 - s["uncertainty_mass"])
            assert s["confidence_score"] == pytest.approx(expected, abs=1e-6)

    def test_scores_bounded(self, binary_dataset):
        X, y = binary_dataset
        clf = DSClassifierMultiQ(num_classes=2, max_iter=MAX_ITER, lossfn="CE")
        clf.fit(X, y, add_single_rules=True, single_rules_breaks=2)
        scores = clf.model.get_rule_uncertainty_scores()

        for s in scores:
            assert 0 <= s["confidence_score"] <= 1
            assert 0 <= s["uncertainty_mass"] <= 1
            assert 0 <= s["max_class_mass"] <= 1

    def test_known_high_uncertainty_rule(self):
        """A rule with high uncertainty mass should have low confidence."""
        model = DSModelMultiQ(k=2)
        # Low-confidence rule: high uncertainty
        model.add_rule(DSRule(lambda x: True, "always_true"), m_sing=[0.1, 0.1], m_uncert=0.8)
        # High-confidence rule: low uncertainty
        model.add_rule(DSRule(lambda x: True, "also_true"), m_sing=[0.8, 0.05], m_uncert=0.15)

        scores = model.get_rule_uncertainty_scores()
        assert scores[0]["confidence_score"] < scores[1]["confidence_score"]
        assert scores[0]["uncertainty_mass"] > scores[1]["uncertainty_mass"]


# ---------- UncertaintyGuidedRefiner ----------

class TestUncertaintyGuidedRefiner:
    def test_refine_increases_rule_count(self, binary_dataset):
        X, y = binary_dataset
        clf = DSClassifierMultiQ(num_classes=2, lr=0.01, max_iter=MAX_ITER, lossfn="CE")
        miner = DecisionTreeMiner(max_depth=2, random_state=SEED)
        refiner = UncertaintyGuidedRefiner(
            clf, miner,
            uncertainty_threshold=0.5,  # aggressive threshold to trigger mining
            max_iterations=1,
            min_covered_samples=5,
        )
        result = refiner.refine(X, y, column_names=["f0", "f1"], single_rules_breaks=2)

        assert result["final_rules"] >= result["initial_rules"]
        assert "history" in result
        assert "clf" in result

    def test_converges_with_low_threshold(self, binary_dataset):
        """With a very low threshold, no rules should be 'weak' and loop should exit immediately."""
        X, y = binary_dataset
        clf = DSClassifierMultiQ(num_classes=2, lr=0.01, max_iter=MAX_ITER, lossfn="CE")
        miner = DecisionTreeMiner(max_depth=2, random_state=SEED)
        refiner = UncertaintyGuidedRefiner(
            clf, miner,
            uncertainty_threshold=0.001,  # very low — nothing should be weak
            max_iterations=3,
            min_covered_samples=5,
        )
        result = refiner.refine(X, y, column_names=["f0", "f1"], single_rules_breaks=2)

        # Should converge immediately (0 or few iterations)
        assert len(result["history"]) <= 1

    def test_history_structure(self, binary_dataset):
        X, y = binary_dataset
        clf = DSClassifierMultiQ(num_classes=2, lr=0.01, max_iter=MAX_ITER, lossfn="CE")
        miner = DecisionTreeMiner(max_depth=2, random_state=SEED)
        refiner = UncertaintyGuidedRefiner(
            clf, miner,
            uncertainty_threshold=0.5,
            max_iterations=1,
            min_covered_samples=5,
        )
        result = refiner.refine(X, y, column_names=["f0", "f1"], single_rules_breaks=2)

        for entry in result["history"]:
            assert "iteration" in entry
            assert "weak_rules_found" in entry
            assert "rules_mined" in entry
            assert "total_rules" in entry

    def test_predictions_valid_after_refinement(self, binary_dataset):
        X, y = binary_dataset
        clf = DSClassifierMultiQ(num_classes=2, lr=0.01, max_iter=MAX_ITER, lossfn="CE")
        miner = DecisionTreeMiner(max_depth=2, random_state=SEED)
        refiner = UncertaintyGuidedRefiner(
            clf, miner,
            uncertainty_threshold=0.5,
            max_iterations=1,
            min_covered_samples=5,
        )
        result = refiner.refine(X, y, column_names=["f0", "f1"], single_rules_breaks=2)

        preds = result["clf"].predict(X)
        assert preds.shape == (len(X),)
        assert set(preds).issubset({0, 1})

        proba = result["clf"].predict_proba(X)
        assert proba.shape == (len(X), 2)
