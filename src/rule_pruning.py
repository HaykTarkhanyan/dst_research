"""
Confidence-Based Rule Pruning for DSGD++.

Prunes rules below an importance threshold using the existing active_rules mechanism.
Computes accuracy-vs-rule-count Pareto frontiers for different pruning strategies.
"""
import copy
import logging

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)


class ConfidenceBasedPruner:
    """Prune DSGD++ rules based on confidence, coverage, or random selection.

    Uses the existing model.active_rules set mechanism in DSModelMultiQ._select_rules
    to filter rules without removing them from the model.

    Parameters
    ----------
    strategy : str
        Pruning strategy: "confidence", "coverage", or "random".
    """

    def __init__(self, strategy="confidence", random_state=509):
        if strategy not in ("confidence", "coverage", "random"):
            raise ValueError(f"Unknown strategy: {strategy}")
        self.strategy = strategy
        self.random_state = random_state

    def _compute_scores(self, clf, X_train=None):
        """Compute per-rule scores based on the selected strategy."""
        n_rules = clf.model.get_rules_size()

        if self.strategy == "confidence":
            rule_scores = clf.model.get_rule_uncertainty_scores()
            return np.array([s["confidence_score"] for s in rule_scores])

        elif self.strategy == "coverage":
            if X_train is None:
                raise ValueError("coverage strategy requires X_train")
            coverage = np.zeros(n_rules)
            for i in range(len(X_train)):
                for j in range(n_rules):
                    if clf.model.preds[j](X_train[i]):
                        coverage[j] += 1
            return coverage / len(X_train)

        elif self.strategy == "random":
            rng = np.random.RandomState(self.random_state)
            return rng.rand(n_rules)

    def prune(self, clf, threshold, X_train=None):
        """Set active_rules to only include rules above the threshold.

        Parameters
        ----------
        clf : DSClassifierMultiQ
            Classifier to prune (modified in-place via active_rules).
        threshold : float
            Minimum score to keep a rule.
        X_train : np.ndarray or None
            Required for "coverage" strategy.

        Returns
        -------
        int : number of active rules after pruning
        """
        scores = self._compute_scores(clf, X_train)
        keep = {i for i in range(len(scores)) if scores[i] >= threshold}
        # Always keep at least 1 rule
        if not keep:
            keep = {int(np.argmax(scores))}
        clf.model.active_rules = keep
        return len(keep)

    def pareto_frontier(self, clf, X_train, y_train, X_test, y_test,
                        thresholds=None, retrain=False, **fit_kwargs):
        """Compute accuracy vs rule count across pruning thresholds.

        Parameters
        ----------
        clf : DSClassifierMultiQ
            Trained classifier.
        X_train, y_train : np.ndarray
            Training data (for retraining and coverage computation).
        X_test, y_test : np.ndarray
            Test data for evaluation.
        thresholds : array-like or None
            Pruning thresholds to sweep. Defaults to 20 points in [0, 0.5].
        retrain : bool
            Whether to retrain after pruning at each threshold.
        **fit_kwargs
            Additional kwargs for retraining.

        Returns
        -------
        list of dicts with keys: threshold, n_active_rules, accuracy, f1
        """
        if thresholds is None:
            thresholds = np.linspace(0, 0.5, 20)

        results = []
        for t in thresholds:
            # Deep copy so each threshold is independent
            clf_copy = copy.deepcopy(clf)
            n_active = self.prune(clf_copy, threshold=t, X_train=X_train)

            if retrain:
                clf_copy.fit(X_train, y_train, **fit_kwargs)

            preds = clf_copy.predict(X_test)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, zero_division=0)

            results.append({
                "threshold": float(t),
                "n_active_rules": n_active,
                "accuracy": acc,
                "f1": f1,
            })

        return results
