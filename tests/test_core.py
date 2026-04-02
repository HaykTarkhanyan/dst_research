import numpy as np
import pytest
import torch

from src.core import (
    belief,
    cls_max,
    cls_pla_max,
    cls_score,
    create_full_uncertainty,
    create_random_maf,
    create_random_maf_k,
    dempster_rule,
    dempster_rule_chain,
    dempster_rule_kt,
    dempster_rule_t,
    plausability,
)

SEED = 509


class TestDempsterRule:
    def test_combining_with_full_uncertainty_is_identity(self):
        m = (0, 0.3, 0.5, 0.2)
        u = (0, 0, 0, 1)
        result = dempster_rule(m, u)
        assert result[0] == 0
        assert pytest.approx(result[1], abs=1e-9) == m[1]
        assert pytest.approx(result[2], abs=1e-9) == m[2]
        assert pytest.approx(result[3], abs=1e-9) == m[3]

    def test_commutativity(self):
        m1 = (0, 0.3, 0.2, 0.5)
        m2 = (0, 0.1, 0.4, 0.5)
        assert dempster_rule(m1, m2) == pytest.approx(dempster_rule(m2, m1), abs=1e-9)

    def test_masses_sum_to_one(self):
        m1 = (0, 0.3, 0.2, 0.5)
        m2 = (0, 0.1, 0.6, 0.3)
        result = dempster_rule(m1, m2)
        assert pytest.approx(sum(result), abs=1e-9) == 1.0

    def test_full_conflict_returns_full_uncertainty(self):
        m1 = (0, 1.0, 0.0, 0.0)
        m2 = (0, 0.0, 1.0, 0.0)
        result = dempster_rule(m1, m2)
        assert result == (0, 0, 0, 1)


class TestDempsterRuleChain:
    def test_single_mass_returns_itself(self):
        m = (0, 0.3, 0.5, 0.2)
        assert dempster_rule_chain(m) == m

    def test_chain_two_equals_pairwise(self):
        m1 = (0, 0.3, 0.2, 0.5)
        m2 = (0, 0.1, 0.4, 0.5)
        assert dempster_rule_chain(m1, m2) == pytest.approx(dempster_rule(m1, m2), abs=1e-9)

    def test_chain_three_associative(self):
        m1 = (0, 0.2, 0.3, 0.5)
        m2 = (0, 0.1, 0.4, 0.5)
        m3 = (0, 0.3, 0.1, 0.6)
        left = dempster_rule(dempster_rule(m1, m2), m3)
        right = dempster_rule_chain(m1, m2, m3)
        assert left == pytest.approx(right, abs=1e-9)


class TestBeliefPlausibility:
    def test_belief_class0(self):
        m = (0, 0.3, 0.5, 0.2)
        assert belief(m, 0) == 0.3

    def test_belief_class1(self):
        m = (0, 0.3, 0.5, 0.2)
        assert belief(m, 1) == 0.5

    def test_plausibility_geq_belief(self):
        m = (0, 0.3, 0.5, 0.2)
        for cls in [0, 1]:
            assert plausability(m, cls) >= belief(m, cls)

    def test_cls_max(self):
        assert cls_max((0, 0.6, 0.3, 0.1)) == 0
        assert cls_max((0, 0.2, 0.7, 0.1)) == 1

    def test_cls_pla_max(self):
        # plaus(cls0) = 1 - m[1] = 0.9, plaus(cls1) = 1 - m[2] = 0.2 → class 0 wins
        assert cls_pla_max((0, 0.1, 0.8, 0.1)) == 0
        # plaus(cls0) = 1 - m[1] = 0.2, plaus(cls1) = 1 - m[2] = 0.9 → class 1 wins
        assert cls_pla_max((0, 0.8, 0.1, 0.1)) == 1

    def test_cls_score_range(self):
        score = cls_score((0, 0.3, 0.5, 0.2))
        assert 0.0 <= score <= 1.0


class TestMAFCreation:
    def test_full_uncertainty(self):
        assert create_full_uncertainty() == (0, 0, 0, 1)

    def test_random_maf_sums_to_one(self):
        m = create_random_maf(0.8)
        assert pytest.approx(sum(m), abs=1e-9) == 1.0
        assert m[0] == 0
        assert m[3] == 0.8

    def test_random_maf_k_shape_and_sum(self):
        np.random.seed(SEED)
        for k in [2, 3, 5]:
            arr = create_random_maf_k(k, uncertainty=0.5)
            assert len(arr) == k + 1
            assert pytest.approx(arr.sum(), abs=1e-6) == 1.0
            assert pytest.approx(arr[-1], abs=1e-6) == 0.5

    def test_random_maf_k_nonnegative(self):
        np.random.seed(SEED)
        arr = create_random_maf_k(4, uncertainty=0.3)
        assert (arr >= 0).all()


class TestDempsterRuleTensor:
    def test_dempster_rule_t_output_shape(self):
        # dempster_rule_t expects indexed elements (scalars from tensor indexing)
        m1 = torch.tensor([[0.3, 0.2, 0.5]])
        m2 = torch.tensor([[0.1, 0.4, 0.5]])
        result = dempster_rule_t(m1[0], m2[0], normalize=True)
        assert result.shape == (3,)

    def test_dempster_rule_t_sums_to_one_normalized(self):
        m1 = torch.tensor([[0.3, 0.2, 0.5]])
        m2 = torch.tensor([[0.1, 0.4, 0.5]])
        result = dempster_rule_t(m1[0], m2[0], normalize=True)
        assert pytest.approx(result.sum().item(), abs=1e-5) == 1.0

    def test_dempster_rule_kt_shape(self):
        m1 = torch.tensor([0.2, 0.1, 0.3, 0.4])
        m2 = torch.tensor([0.3, 0.1, 0.2, 0.4])
        result = dempster_rule_kt(m1, m2)
        assert result.shape == m1.shape

    def test_dempster_rule_kt_normalized_sums_to_one(self):
        m1 = torch.tensor([0.2, 0.1, 0.3, 0.4])
        m2 = torch.tensor([0.3, 0.1, 0.2, 0.4])
        result = dempster_rule_kt(m1, m2, normalize=True)
        assert pytest.approx(result.sum().item(), abs=1e-5) == 1.0
