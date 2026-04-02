import numpy as np
import pytest
import torch

from src.DSModelMultiQ import DSModelMultiQ
from src.DSRule import DSRule

SEED = 509


class TestDSModelMultiQInit:
    def test_empty_model(self):
        model = DSModelMultiQ(k=2)
        assert model.n == 0
        assert model.k == 2
        assert len(model.preds) == 0

    def test_add_rule_increments_count(self):
        np.random.seed(SEED)
        model = DSModelMultiQ(k=2)
        rule = DSRule(lambda x: x[0] > 0, "X[0] > 0")
        model.add_rule(rule, method="random")
        assert model.n == 1
        assert len(model._params) == 1


class TestDSModelMultiQNormalize:
    def test_masses_nonnegative_after_normalize(self):
        model = DSModelMultiQ(k=2)
        model._params = [torch.tensor([-0.1, 0.5, 0.8], requires_grad=False)]
        model.normalize()
        for p in model._params:
            assert (p >= 0).all()

    def test_masses_sum_leq_one_after_normalize(self):
        model = DSModelMultiQ(k=3)
        model._params = [torch.tensor([0.5, 0.3, 0.4, 0.2], requires_grad=False)]
        model.normalize()
        for p in model._params:
            assert p.sum().item() <= 1.0 + 1e-6

    def test_normalize_adds_uncertainty_when_sum_lt_one(self):
        model = DSModelMultiQ(k=2)
        model._params = [torch.tensor([0.1, 0.1, 0.1], requires_grad=False)]
        model.normalize()
        p = model._params[0]
        assert pytest.approx(p.sum().item(), abs=1e-6) == 1.0
        assert p[-1].item() >= 0.1


class TestDSModelMultiQForward:
    @pytest.fixture(scope="class")
    def model_and_data(self):
        np.random.seed(SEED)
        X = np.random.RandomState(SEED).randn(6, 2)
        model = DSModelMultiQ(k=2)
        model.generate_statistic_single_rules(X, breaks=2)
        X_idx = np.insert(X, 0, values=np.arange(len(X)), axis=1)
        Xt = torch.Tensor(X_idx)
        out = model(Xt)
        return model, X, Xt, out

    def test_output_shape(self, model_and_data):
        _, X, _, out = model_and_data
        assert out.shape == (len(X), 2)

    def test_output_sums_to_one(self, model_and_data):
        _, _, _, out = model_and_data
        for s in out.sum(dim=1):
            assert pytest.approx(s.item(), abs=1e-4) == 1.0

    def test_output_nonnegative(self, model_and_data):
        _, _, _, out = model_and_data
        assert (out >= 0).all()


class TestRuleGeneration:
    def test_statistic_single_rules_count(self):
        np.random.seed(SEED)
        model = DSModelMultiQ(k=2)
        model.generate_statistic_single_rules(np.random.randn(8, 3), breaks=2)
        assert model.n == 9

    def test_statistic_single_rules_with_in_between(self):
        np.random.seed(SEED)
        model = DSModelMultiQ(k=2, add_in_between_rules=True)
        model.generate_statistic_single_rules(np.random.randn(8, 3), breaks=2)
        assert model.n == 15

    def test_mult_pair_rules_count(self):
        np.random.seed(SEED)
        model = DSModelMultiQ(k=2)
        model.generate_mult_pair_rules(np.random.randn(8, 3))
        assert model.n == 6

    def test_generated_rules_fire_correctly(self):
        rng = np.random.RandomState(SEED)
        X = rng.randn(10, 2)
        model = DSModelMultiQ(k=2)
        model.generate_statistic_single_rules(X, breaks=2)
        fired_count = 0
        for i in range(len(X)):
            sel = model._select_rules(torch.Tensor(X[i]))
            if len(sel) > 0:
                fired_count += 1
        # at least most samples should have rules fire
        assert fired_count >= len(X) // 2


class TestFindMostImportantRules:
    def test_returns_dict_with_class_keys(self):
        np.random.seed(SEED)
        model = DSModelMultiQ(k=2)
        model.generate_statistic_single_rules(np.random.randn(6, 2), breaks=2)
        rules = model.find_most_important_rules()
        assert 0 in rules and 1 in rules

    def test_scores_are_nonnegative(self):
        np.random.seed(SEED)
        model = DSModelMultiQ(k=2)
        model.generate_statistic_single_rules(np.random.randn(6, 2), breaks=2)
        for cls_rules in model.find_most_important_rules(threshold=0.0).values():
            for score, *_ in cls_rules:
                assert score >= 0


class TestGetRulesSize:
    def test_rules_size(self):
        np.random.seed(SEED)
        model = DSModelMultiQ(k=2)
        assert model.get_rules_size() == 0
        model.add_rule(DSRule(lambda x: True, "always"), method="random")
        assert model.get_rules_size() == 1
