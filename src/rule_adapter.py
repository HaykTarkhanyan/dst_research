"""
Rule Adapter: converts rule strings from external miners into DSRule objects.

Supports rule formats from imodels (SkopeRules, RuleFit), wittgenstein (RIPPER),
and sklearn DecisionTreeClassifier leaf paths.
"""
import logging
import operator
import re
from dataclasses import dataclass

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from .DSRule import DSRule

logger = logging.getLogger(__name__)

OP_MAP = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}

# Regex: feature name (greedy minimal), operator, numeric value
# Operator order matters: <= before <, >= before >
CONDITION_PATTERN = re.compile(
    r"(.+?)\s*(<=|>=|!=|<|>|==)\s*(-?[\d]+\.?[\d]*(?:[eE][+-]?\d+)?)"
)

# Splitters for conjunctive rules from different sources
CONJUNCTION_SPLITTERS = re.compile(r"\s+and\s+|\s*&\s*|\s*\^\s*", re.IGNORECASE)


@dataclass
class RuleCondition:
    feature_index: int
    feature_name: str
    op_str: str
    op_func: object
    value: float


def build_feature_name_map(column_names):
    """Build mapping from feature name strings to column indices.

    Handles dataset column names, "X[i]" default format (from DSModelMultiQ),
    and "feature_i" format (from sklearn miners on numpy arrays).
    """
    name_map = {}
    for i, name in enumerate(column_names):
        name_map[name] = i
        name_map[name.strip()] = i
        # Also register positional aliases
        name_map[f"feature_{i}"] = i
        name_map[f"X[{i}]"] = i
        name_map[f"x[{i}]"] = i
        name_map[f"feature{i}"] = i
    return name_map


def parse_condition(condition_str, feature_name_map):
    """Parse a single condition string like 'feature_0 > 5.0' into a RuleCondition."""
    condition_str = condition_str.strip()
    match = CONDITION_PATTERN.match(condition_str)
    if match is None:
        raise ValueError(f"Cannot parse condition: '{condition_str}'")

    feature_name = match.group(1).strip()
    op_str = match.group(2)
    value = float(match.group(3))

    if feature_name not in feature_name_map:
        raise KeyError(f"Unknown feature '{feature_name}'. "
                       f"Known features: {list(feature_name_map.keys())[:10]}...")

    return RuleCondition(
        feature_index=feature_name_map[feature_name],
        feature_name=feature_name,
        op_str=op_str,
        op_func=OP_MAP[op_str],
        value=value,
    )


def parse_rule_string(rule_str, feature_name_map):
    """Parse a conjunctive rule string into a list of RuleConditions.

    Handles 'and', '&', '^' as conjunction operators (case-insensitive).
    """
    parts = CONJUNCTION_SPLITTERS.split(rule_str.strip())
    return [parse_condition(part, feature_name_map) for part in parts if part.strip()]


def conditions_to_dsrule(conditions, caption=None):
    """Convert a list of RuleConditions into a DSRule with a callable lambda.

    The lambda captures condition checks via default arguments to avoid
    closure issues (same pattern as DSModelMultiQ.generate_statistic_single_rules).
    """
    if caption is None:
        caption = " and ".join(
            f"{c.feature_name} {c.op_str} {c.value}" for c in conditions
        )

    checks = tuple((c.feature_index, c.op_func, c.value) for c in conditions)
    return DSRule(
        lambda x, _checks=checks: all(op(x[i], v) for i, op, v in _checks),
        caption,
    )


def convert_rule_string(rule_str, feature_name_map):
    """End-to-end: parse a rule string and return a DSRule."""
    conditions = parse_rule_string(rule_str, feature_name_map)
    return conditions_to_dsrule(conditions, caption=rule_str.strip())


# ---------------------------------------------------------------------------
# Source-specific rule extraction
# ---------------------------------------------------------------------------

def extract_rules_from_skope(clf, feature_name_map, max_rules=None):
    """Extract DSRules from a fitted imodels.SkopeRulesClassifier.

    clf.rules_ is a list of imodels.util.rule.Rule objects.
    Each has str(rule) for the rule string and rule.args = (precision, recall, n_samples).
    Returns rules sorted by precision (descending).
    """
    if not hasattr(clf, "rules_") or not clf.rules_:
        return []

    rules_with_meta = []
    for item in clf.rules_:
        rule_str = str(item)
        # Rule.args = (precision, recall, n_samples) or may be a tuple/other
        precision = 0.0
        if hasattr(item, "args") and item.args:
            precision = item.args[0]
        rules_with_meta.append((rule_str, precision))

    # Sort by precision descending
    rules_with_meta.sort(key=lambda x: x[1], reverse=True)

    if max_rules is not None:
        rules_with_meta = rules_with_meta[:max_rules]

    ds_rules = []
    for rule_str, _prec in rules_with_meta:
        try:
            ds_rules.append(convert_rule_string(rule_str, feature_name_map))
        except (ValueError, KeyError) as e:
            logger.warning("Skipping unparseable SkopeRules rule '%s': %s", rule_str, e)
    return ds_rules


def extract_rules_from_rulefit(clf, feature_name_map, max_rules=None, min_importance=0.01):
    """Extract DSRules from a fitted imodels.RuleFitClassifier.

    clf.get_rules() returns a DataFrame with columns: rule, type, coef, importance, support.
    Filters to type='rule' with non-zero coefficient.
    """
    rules_df = clf.get_rules()
    # Keep only actual rules (not linear terms) with nonzero coefficients
    mask = (rules_df["type"] == "rule") & (rules_df["coef"].abs() > 0)
    if min_importance > 0:
        mask = mask & (rules_df["importance"] >= min_importance)
    rules_df = rules_df[mask].sort_values("importance", ascending=False)

    if max_rules is not None:
        rules_df = rules_df.head(max_rules)

    ds_rules = []
    for rule_str in rules_df["rule"]:
        try:
            ds_rules.append(convert_rule_string(rule_str, feature_name_map))
        except (ValueError, KeyError) as e:
            logger.warning("Skipping unparseable RuleFit rule '%s': %s", rule_str, e)
    return ds_rules


def _parse_ripper_cond(cond, feature_name_map):
    """Parse a wittgenstein Cond object into RuleCondition(s).

    Cond has .feature (str) and .val (str) where val can be:
      ">2.84"       -> feature > 2.84
      "<0.5"        -> feature < 0.5
      ">=1.0"       -> feature >= 1.0
      "<=3.0"       -> feature <= 3.0
      "1.81 - 2.34" -> 1.81 <= feature < 2.34 (range, returns 2 conditions)
      "category"    -> feature == category (categorical)
    """
    feat_name = str(cond.feature).strip()
    val_str = str(cond.val).strip()

    if feat_name not in feature_name_map:
        raise KeyError(f"Unknown feature '{feat_name}'")
    feat_idx = feature_name_map[feat_name]

    # Range: "1.81 - 2.34" or "1.81-2.34"
    range_match = re.match(r"(-?[\d.eE+-]+)\s*-\s*(-?[\d.eE+-]+)$", val_str)
    if range_match:
        lo = float(range_match.group(1))
        hi = float(range_match.group(2))
        return [
            RuleCondition(feat_idx, feat_name, ">=", operator.ge, lo),
            RuleCondition(feat_idx, feat_name, "<", operator.lt, hi),
        ]

    # Comparison: ">2.84", ">=1.0", "<0.5", "<=3.0"
    comp_match = re.match(r"(<=|>=|<|>)\s*(-?[\d.eE+-]+)$", val_str)
    if comp_match:
        op_str = comp_match.group(1)
        value = float(comp_match.group(2))
        return [RuleCondition(feat_idx, feat_name, op_str, OP_MAP[op_str], value)]

    # Categorical: exact match
    try:
        value = float(val_str)
        return [RuleCondition(feat_idx, feat_name, "==", operator.eq, value)]
    except ValueError:
        logger.warning("Skipping categorical RIPPER condition: %s=%s", feat_name, val_str)
        return []


def extract_rules_from_ripper(clf, feature_name_map, max_rules=None):
    """Extract DSRules from a fitted wittgenstein.RIPPER classifier.

    Iterates over clf.ruleset_ Rule objects. Each Rule has .conds (list of Cond).
    Each Cond has .feature and .val attributes.
    """
    ds_rules = []

    try:
        for rule in clf.ruleset_:
            all_conditions = []
            skip = False
            for cond in rule.conds:
                try:
                    all_conditions.extend(_parse_ripper_cond(cond, feature_name_map))
                except (KeyError, ValueError) as e:
                    logger.warning("Skipping RIPPER rule with unparseable cond '%s': %s", cond, e)
                    skip = True
                    break
            if skip or not all_conditions:
                continue
            caption = " and ".join(f"{c.feature_name} {c.op_str} {c.value}" for c in all_conditions)
            ds_rules.append(conditions_to_dsrule(all_conditions, caption))
    except Exception as e:
        logger.warning("Failed to extract RIPPER rules: %s", e)

    if max_rules is not None:
        ds_rules = ds_rules[:max_rules]
    return ds_rules


def extract_rules_from_tree(tree_clf, feature_names, feature_name_map=None, max_depth=None):
    """Extract conjunctive rules from DecisionTreeClassifier leaf paths.

    Each root-to-leaf path becomes a conjunctive rule. Only extracts paths
    where the leaf has a dominant class (purity > 0.5).
    """
    tree = tree_clf.tree_
    ds_rules = []

    def _traverse(node, conditions):
        # Leaf node
        if tree.children_left[node] == tree.children_right[node]:
            class_counts = tree.value[node][0]
            total = class_counts.sum()
            if total == 0:
                return
            dominant_class = np.argmax(class_counts)
            purity = class_counts[dominant_class] / total
            if purity > 0.5 and len(conditions) > 0:
                ds_rules.append(conditions_to_dsrule(list(conditions)))
            return

        if max_depth is not None and len(conditions) >= max_depth:
            return

        feat_idx = tree.feature[node]
        threshold = tree.threshold[node]
        feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"feature_{feat_idx}"

        # Left child: feature <= threshold
        left_cond = RuleCondition(feat_idx, feat_name, "<=", operator.le, threshold)
        _traverse(tree.children_left[node], conditions + [left_cond])

        # Right child: feature > threshold
        right_cond = RuleCondition(feat_idx, feat_name, ">", operator.gt, threshold)
        _traverse(tree.children_right[node], conditions + [right_cond])

    _traverse(0, [])
    return ds_rules
