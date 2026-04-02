"""
Multi-Source Rule Ensemble for DSGD++.

Combines rules from multiple mining sources into a single DSGD++ model.
Dempster's rule naturally handles heterogeneous evidence sources.
"""
import logging

import numpy as np

from .DSClassifierMultiQ import DSClassifierMultiQ
from .rule_adapter import build_feature_name_map

logger = logging.getLogger(__name__)


class MultiSourceEnsemble:
    """Combine rules from multiple miners into one DSGD++ model.

    Parameters
    ----------
    miners : list[RuleMiner]
        List of rule mining wrappers to generate rules from.
    num_classes : int
        Number of target classes.
    base_breaks : int
        Number of breaks for standard single-feature rules.
    add_mult_rules : bool
        Whether to include multiplication pair rules.
    maf_method : str
        MAF initialization method for mined rules ("random", "uniform").
    **clf_kwargs
        Additional kwargs passed to DSClassifierMultiQ constructor.
    """

    def __init__(self, miners, num_classes=2, base_breaks=2,
                 add_mult_rules=False, maf_method="random", **clf_kwargs):
        self.miners = miners
        self.num_classes = num_classes
        self.base_breaks = base_breaks
        self.add_mult_rules = add_mult_rules
        self.maf_method = maf_method
        self.clf_kwargs = clf_kwargs
        self.clf = None
        self.rule_sources_ = {}

    def fit(self, X_train, y_train, column_names=None, **fit_kwargs):
        """Fit the ensemble: generate rules from all sources, then train DSGD++.

        Returns
        -------
        self
        """
        if column_names is None:
            column_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        feature_name_map = build_feature_name_map(column_names)

        self.clf = DSClassifierMultiQ(
            num_classes=self.num_classes,
            maf_method=self.maf_method,
            **self.clf_kwargs,
        )

        # Add standard single-feature rules
        self.clf.model.generate_statistic_single_rules(
            X_train, breaks=self.base_breaks, column_names=column_names,
        )
        if self.add_mult_rules:
            self.clf.model.generate_mult_pair_rules(X_train, column_names=column_names)

        base_rules = self.clf.model.get_rules_size()
        self.rule_sources_["base"] = base_rules
        logger.info("Base rules (single-feature): %d", base_rules)

        # Add rules from each miner
        for miner in self.miners:
            miner_name = type(miner).__name__
            try:
                miner.fit(X_train, y_train, feature_names=column_names)
                new_rules = miner.extract_rules(feature_name_map)
                for rule in new_rules:
                    self.clf.model.add_rule(rule, method=self.maf_method)
                self.rule_sources_[miner_name] = len(new_rules)
                logger.info("%s: %d rules mined", miner_name, len(new_rules))
            except Exception as e:
                logger.warning("Miner %s failed: %s", miner_name, e)
                self.rule_sources_[miner_name] = 0

        total = self.clf.model.get_rules_size()
        logger.info("Total rules (all sources): %d", total)

        # Train DSGD++ on combined rule set
        self.clf.fit(X_train, y_train, **fit_kwargs)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def get_rule_sources(self):
        """Return dict of {source_name: rule_count}."""
        return dict(self.rule_sources_)
