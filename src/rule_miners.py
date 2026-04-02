"""
Thin sklearn-compatible wrappers around external rule mining algorithms.

Each miner exposes fit(X, y, feature_names) and extract_rules(feature_name_map) -> list[DSRule].
"""
import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from .rule_adapter import (
    build_feature_name_map,
    extract_rules_from_rulefit,
    extract_rules_from_ripper,
    extract_rules_from_skope,
    extract_rules_from_tree,
)

logger = logging.getLogger(__name__)


class RuleMiner(ABC):
    """Abstract base class for rule mining wrappers."""

    @abstractmethod
    def fit(self, X, y, feature_names=None):
        """Fit the miner on training data."""

    @abstractmethod
    def extract_rules(self, feature_name_map):
        """Extract mined rules as DSRule objects."""

    def fit_extract(self, X, y, feature_names=None, feature_name_map=None):
        """Convenience: fit and extract in one call."""
        self.fit(X, y, feature_names=feature_names)
        if feature_name_map is None:
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            feature_name_map = build_feature_name_map(feature_names)
        return self.extract_rules(feature_name_map)


class SkopeRulesMiner(RuleMiner):
    """Wrapper around imodels.SkopeRulesClassifier."""

    def __init__(self, n_estimators=30, max_depth=3, precision_min=0.3,
                 recall_min=0.1, max_rules=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.precision_min = precision_min
        self.recall_min = recall_min
        self.max_rules = max_rules
        self.clf_ = None
        self.feature_names_ = None

    def fit(self, X, y, feature_names=None):
        from imodels import SkopeRulesClassifier
        self.feature_names_ = feature_names
        self.clf_ = SkopeRulesClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            precision_min=self.precision_min,
            recall_min=self.recall_min,
        )
        self.clf_.fit(X, y, feature_names=feature_names)
        return self

    def extract_rules(self, feature_name_map):
        return extract_rules_from_skope(self.clf_, feature_name_map, max_rules=self.max_rules)


class RipperMiner(RuleMiner):
    """Wrapper around wittgenstein.RIPPER."""

    def __init__(self, prune_size=0.33, k=2, max_rules=None, pos_class=None):
        self.prune_size = prune_size
        self.k = k
        self.max_rules = max_rules
        self.pos_class = pos_class
        self.clf_ = None
        self.feature_names_ = None

    def fit(self, X, y, feature_names=None):
        import wittgenstein as lw
        self.feature_names_ = feature_names
        # RIPPER works best with DataFrames
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df["_target"] = y
        # Auto-detect positive class: use the minority class (or 1 if balanced)
        pos_class = self.pos_class
        if pos_class is None:
            unique, counts = np.unique(y, return_counts=True)
            pos_class = unique[np.argmin(counts)]
        self.clf_ = lw.RIPPER(prune_size=self.prune_size, k=self.k)
        self.clf_.fit(df, class_feat="_target", pos_class=pos_class)
        return self

    def extract_rules(self, feature_name_map):
        return extract_rules_from_ripper(self.clf_, feature_name_map, max_rules=self.max_rules)


class RuleFitMiner(RuleMiner):
    """Wrapper around imodels.RuleFitClassifier."""

    def __init__(self, max_rules=20, tree_size=4, max_extracted_rules=None,
                 min_importance=0.01):
        self.max_rules = max_rules
        self.tree_size = tree_size
        self.max_extracted_rules = max_extracted_rules
        self.min_importance = min_importance
        self.clf_ = None
        self.feature_names_ = None

    def fit(self, X, y, feature_names=None):
        from imodels import RuleFitClassifier
        self.feature_names_ = feature_names
        self.clf_ = RuleFitClassifier(
            max_rules=self.max_rules,
            tree_size=self.tree_size,
        )
        self.clf_.fit(X, y, feature_names=feature_names)
        return self

    def extract_rules(self, feature_name_map):
        return extract_rules_from_rulefit(
            self.clf_, feature_name_map,
            max_rules=self.max_extracted_rules,
            min_importance=self.min_importance,
        )


class DecisionTreeMiner(RuleMiner):
    """Extract rules from DecisionTreeClassifier leaf paths."""

    def __init__(self, max_depth=4, random_state=509, max_rules=None):
        self.max_depth = max_depth
        self.random_state = random_state
        self.max_rules = max_rules
        self.clf_ = None
        self.feature_names_ = None

    def fit(self, X, y, feature_names=None):
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        self.feature_names_ = feature_names
        self.clf_ = DecisionTreeClassifier(
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self.clf_.fit(X, y)
        return self

    def extract_rules(self, feature_name_map):
        rules = extract_rules_from_tree(
            self.clf_, self.feature_names_, feature_name_map,
            max_depth=self.max_depth,
        )
        if self.max_rules is not None:
            rules = rules[:self.max_rules]
        return rules
