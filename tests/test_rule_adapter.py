"""Tests for rule_adapter: parsing, DSRule construction, source-specific extraction."""
import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.rule_adapter import (
    RuleCondition,
    build_feature_name_map,
    conditions_to_dsrule,
    convert_rule_string,
    extract_rules_from_tree,
    parse_condition,
    parse_rule_string,
)

FEATURE_NAMES = ["age", "income", "score"]
FEATURE_MAP = build_feature_name_map(FEATURE_NAMES)


# ---------- build_feature_name_map ----------

class TestBuildFeatureNameMap:
    def test_dataset_column_names(self):
        fm = build_feature_name_map(["age", "income"])
        assert fm["age"] == 0
        assert fm["income"] == 1

    def test_positional_aliases(self):
        fm = build_feature_name_map(["age", "income"])
        assert fm["feature_0"] == 0
        assert fm["feature_1"] == 1
        assert fm["X[0]"] == 0
        assert fm["X[1]"] == 1
        assert fm["feature0"] == 0

    def test_strips_whitespace(self):
        fm = build_feature_name_map(["age ", " income"])
        assert fm["age"] == 0
        assert fm["income"] == 1


# ---------- parse_condition ----------

class TestParseCondition:
    def test_greater_than(self):
        c = parse_condition("age > 5.0", FEATURE_MAP)
        assert c.feature_index == 0
        assert c.op_str == ">"
        assert c.value == 5.0

    def test_less_equal(self):
        c = parse_condition("income <= 3.2", FEATURE_MAP)
        assert c.feature_index == 1
        assert c.op_str == "<="
        assert c.value == 3.2

    def test_greater_equal(self):
        c = parse_condition("score >= 0.5", FEATURE_MAP)
        assert c.feature_index == 2
        assert c.op_str == ">="
        assert c.value == 0.5

    def test_less_than(self):
        c = parse_condition("age < 10", FEATURE_MAP)
        assert c.op_str == "<"
        assert c.value == 10.0

    def test_equal(self):
        c = parse_condition("age == 1", FEATURE_MAP)
        assert c.op_str == "=="
        assert c.value == 1.0

    def test_not_equal(self):
        c = parse_condition("age != 0", FEATURE_MAP)
        assert c.op_str == "!="
        assert c.value == 0.0

    def test_negative_value(self):
        c = parse_condition("score > -1.5", FEATURE_MAP)
        assert c.value == -1.5

    def test_scientific_notation(self):
        c = parse_condition("age > 1.5e-3", FEATURE_MAP)
        assert c.value == pytest.approx(0.0015)

    def test_feature_alias(self):
        c = parse_condition("feature_0 > 5", FEATURE_MAP)
        assert c.feature_index == 0

    def test_unknown_feature_raises(self):
        with pytest.raises(KeyError, match="Unknown feature"):
            parse_condition("unknown_col > 5", FEATURE_MAP)

    def test_unparseable_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_condition("not a valid condition", FEATURE_MAP)


# ---------- parse_rule_string ----------

class TestParseRuleString:
    def test_single_condition(self):
        conditions = parse_rule_string("age > 5", FEATURE_MAP)
        assert len(conditions) == 1
        assert conditions[0].feature_name == "age"

    def test_and_conjunction(self):
        conditions = parse_rule_string("age > 5 and income <= 3.2", FEATURE_MAP)
        assert len(conditions) == 2
        assert conditions[0].feature_name == "age"
        assert conditions[1].feature_name == "income"

    def test_ampersand_conjunction(self):
        conditions = parse_rule_string("age > 5 & income <= 3.2", FEATURE_MAP)
        assert len(conditions) == 2

    def test_caret_conjunction(self):
        conditions = parse_rule_string("age > 5 ^ income <= 3.2", FEATURE_MAP)
        assert len(conditions) == 2

    def test_case_insensitive_and(self):
        conditions = parse_rule_string("age > 5 AND income <= 3", FEATURE_MAP)
        assert len(conditions) == 2

    def test_three_conditions(self):
        conditions = parse_rule_string("age > 5 and income <= 3 and score >= 0.5", FEATURE_MAP)
        assert len(conditions) == 3


# ---------- conditions_to_dsrule ----------

class TestConditionsToDSRule:
    def test_single_condition_fires(self):
        cond = [RuleCondition(0, "age", ">", lambda a, b: a > b, 5.0)]
        rule = conditions_to_dsrule(cond, "age > 5")
        x = np.array([6.0, 0.0, 0.0])
        assert rule(x) is True

    def test_single_condition_does_not_fire(self):
        cond = [RuleCondition(0, "age", ">", lambda a, b: a > b, 5.0)]
        rule = conditions_to_dsrule(cond, "age > 5")
        x = np.array([4.0, 0.0, 0.0])
        assert rule(x) is False

    def test_conjunctive_rule_both_true(self):
        conditions = parse_rule_string("age > 5 and income <= 3", FEATURE_MAP)
        rule = conditions_to_dsrule(conditions)
        x = np.array([6.0, 2.0, 0.0])
        assert rule(x) is True

    def test_conjunctive_rule_one_false(self):
        conditions = parse_rule_string("age > 5 and income <= 3", FEATURE_MAP)
        rule = conditions_to_dsrule(conditions)
        x = np.array([6.0, 4.0, 0.0])
        assert rule(x) is False

    def test_caption_auto_generated(self):
        conditions = parse_rule_string("age > 5", FEATURE_MAP)
        rule = conditions_to_dsrule(conditions)
        assert "age" in str(rule)

    def test_caption_explicit(self):
        conditions = parse_rule_string("age > 5", FEATURE_MAP)
        rule = conditions_to_dsrule(conditions, caption="custom caption")
        assert str(rule) == "custom caption"


# ---------- convert_rule_string (end-to-end) ----------

class TestConvertRuleString:
    def test_single_condition(self):
        rule = convert_rule_string("age > 5", FEATURE_MAP)
        assert rule(np.array([6.0, 0.0, 0.0])) is True
        assert rule(np.array([4.0, 0.0, 0.0])) is False

    def test_conjunctive(self):
        rule = convert_rule_string("age > 5 and income <= 3", FEATURE_MAP)
        assert rule(np.array([6.0, 2.0, 0.0])) is True
        assert rule(np.array([6.0, 4.0, 0.0])) is False
        assert rule(np.array([4.0, 2.0, 0.0])) is False

    def test_preserves_caption(self):
        rule = convert_rule_string("  age > 5 and income <= 3  ", FEATURE_MAP)
        assert str(rule) == "age > 5 and income <= 3"


# ---------- extract_rules_from_tree ----------

class TestExtractRulesFromTree:
    @pytest.fixture
    def simple_tree(self):
        """A shallow decision tree on a trivially separable dataset."""
        rng = np.random.RandomState(509)
        X = np.vstack([rng.randn(20, 2) + [-2, -2], rng.randn(20, 2) + [2, 2]])
        y = np.array([0] * 20 + [1] * 20)
        tree = DecisionTreeClassifier(max_depth=2, random_state=509)
        tree.fit(X, y)
        return tree, ["feat_a", "feat_b"]

    def test_returns_dsrules(self, simple_tree):
        tree, names = simple_tree
        fm = build_feature_name_map(names)
        rules = extract_rules_from_tree(tree, names, fm)
        assert len(rules) > 0
        for r in rules:
            assert callable(r)
            assert len(str(r)) > 0

    def test_rules_fire_correctly(self, simple_tree):
        tree, names = simple_tree
        fm = build_feature_name_map(names)
        rules = extract_rules_from_tree(tree, names, fm)
        # At least one rule should fire on a point from each cluster
        x_class0 = np.array([-2.0, -2.0])
        x_class1 = np.array([2.0, 2.0])
        fired_0 = [r for r in rules if r(x_class0)]
        fired_1 = [r for r in rules if r(x_class1)]
        assert len(fired_0) > 0
        assert len(fired_1) > 0

    def test_max_depth_limits_conditions(self, simple_tree):
        tree, names = simple_tree
        fm = build_feature_name_map(names)
        rules = extract_rules_from_tree(tree, names, fm, max_depth=1)
        # With max_depth=1, each rule has at most 1 condition
        for r in rules:
            # Caption has at most one 'and'
            assert " and " not in str(r)
