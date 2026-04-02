"""
Uncertainty-Guided Iterative Rule Mining for DSGD++.

The core algorithm: train DSGD++ -> identify high-uncertainty rules ->
mine targeted conjunctive rules for ambiguous data regions -> retrain.
"""
import logging

import numpy as np

from .rule_adapter import build_feature_name_map

logger = logging.getLogger(__name__)


class UncertaintyGuidedRefiner:
    """Iteratively refines a DSGD++ classifier by mining rules for uncertain regions.

    Parameters
    ----------
    clf : DSClassifierMultiQ
        The classifier to refine. Will be modified in-place.
    miner : RuleMiner
        External rule miner (SkopeRulesMiner, RipperMiner, etc.).
    uncertainty_threshold : float
        Rules with confidence_score below this are considered "weak".
        confidence_score = max(class_masses) * (1 - uncertainty_mass).
    max_iterations : int
        Maximum refinement iterations.
    min_covered_samples : int
        Minimum samples a weak rule must cover to trigger mining.
    max_rules_per_weak : int or None
        Cap on new rules mined per weak rule (prevents explosion).
    max_total_rules : int or None
        Hard cap on total model rules. Stops adding once reached.
    max_weak_to_refine : int or None
        Max number of weak rules to refine per iteration (picks the weakest first).
    """

    def __init__(self, clf, miner, uncertainty_threshold=0.3, max_iterations=5,
                 min_covered_samples=20, max_rules_per_weak=3,
                 max_total_rules=100, max_weak_to_refine=10):
        self.clf = clf
        self.miner = miner
        self.uncertainty_threshold = uncertainty_threshold
        self.max_iterations = max_iterations
        self.min_covered_samples = min_covered_samples
        self.max_rules_per_weak = max_rules_per_weak
        self.max_total_rules = max_total_rules
        self.max_weak_to_refine = max_weak_to_refine

    def refine(self, X_train, y_train, column_names=None,
               single_rules_breaks=2, add_mult_rules=False, **fit_kwargs):
        """Run the iterative uncertainty-guided refinement loop.

        Parameters
        ----------
        X_train : np.ndarray
            Training features.
        y_train : np.ndarray
            Training labels.
        column_names : list[str] or None
            Feature column names. If None, defaults to feature_0, feature_1, ...
        single_rules_breaks : int
            Number of breaks for initial single-feature rules.
        add_mult_rules : bool
            Whether to add multiplication pair rules in the initial fit.
        **fit_kwargs
            Additional kwargs passed to clf.fit() and retrain calls.

        Returns
        -------
        dict with keys:
            - "clf": the refined classifier
            - "history": list of per-iteration metrics
        """
        if column_names is None:
            column_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        feature_name_map = build_feature_name_map(column_names)

        # Step 1: Initial training with standard single-feature rules
        logger.info("Initial DSGD++ training with %d-break single-feature rules", single_rules_breaks)
        self.clf.fit(
            X_train, y_train,
            add_single_rules=True,
            single_rules_breaks=single_rules_breaks,
            add_mult_rules=add_mult_rules,
            column_names=column_names,
            **fit_kwargs,
        )
        initial_rules = self.clf.model.get_rules_size()
        logger.info("Initial model: %d rules", initial_rules)

        history = []

        for iteration in range(self.max_iterations):
            # Step 2a: Identify high-uncertainty rules
            scores = self.clf.model.get_rule_uncertainty_scores()
            weak_rules = [s for s in scores if s["confidence_score"] < self.uncertainty_threshold]

            logger.info("Iteration %d: %d/%d rules are weak (confidence < %.2f)",
                        iteration + 1, len(weak_rules), len(scores), self.uncertainty_threshold)

            if not weak_rules:
                logger.info("No weak rules found — converged")
                break

            # Check global rule cap
            current_rules = self.clf.model.get_rules_size()
            if self.max_total_rules and current_rules >= self.max_total_rules:
                logger.info("Total rules (%d) reached cap (%d) — stopping",
                            current_rules, self.max_total_rules)
                break

            # Sort by confidence (weakest first) and limit how many we refine
            weak_rules.sort(key=lambda s: s["confidence_score"])
            if self.max_weak_to_refine is not None:
                weak_rules = weak_rules[:self.max_weak_to_refine]

            # Step 2d: Mine targeted rules for each weak rule's covered region
            total_new_rules = 0
            for weak in weak_rules:
                # Check global cap before each mining step
                if self.max_total_rules and self.clf.model.get_rules_size() >= self.max_total_rules:
                    logger.info("  Hit total rule cap (%d) mid-iteration", self.max_total_rules)
                    break

                pred = weak["pred"]
                rule_idx = weak["rule_idx"]

                # Find covered samples
                covered_mask = np.array([bool(pred(X_train[i])) for i in range(len(X_train))])
                n_covered = covered_mask.sum()

                if n_covered < self.min_covered_samples:
                    continue

                X_covered = X_train[covered_mask]
                y_covered = y_train[covered_mask]

                # Skip pure regions (nothing to disambiguate)
                if len(np.unique(y_covered)) < 2:
                    continue

                # Mine rules on the covered subset
                try:
                    self.miner.fit(X_covered, y_covered, feature_names=column_names)
                    new_rules = self.miner.extract_rules(feature_name_map)
                except Exception as e:
                    logger.warning("Mining failed for rule %d ('%s'): %s",
                                   rule_idx, weak["caption"], e)
                    continue

                # Cap rules per weak rule
                if self.max_rules_per_weak is not None:
                    new_rules = new_rules[:self.max_rules_per_weak]

                # Don't exceed global cap
                if self.max_total_rules:
                    room = self.max_total_rules - self.clf.model.get_rules_size()
                    new_rules = new_rules[:room]

                for dsrule in new_rules:
                    self.clf.model.add_rule(dsrule, method="random")
                    total_new_rules += 1

            if total_new_rules == 0:
                logger.info("No new rules mined — converged")
                break

            # Step 2f: Retrain with expanded rule set
            logger.info("Added %d new rules, retraining (%d total rules)",
                        total_new_rules, self.clf.model.get_rules_size())
            self.clf.fit(X_train, y_train, **fit_kwargs)

            history.append({
                "iteration": iteration + 1,
                "weak_rules_found": len(weak_rules),
                "rules_mined": total_new_rules,
                "total_rules": self.clf.model.get_rules_size(),
            })

        return {
            "clf": self.clf,
            "history": history,
            "initial_rules": initial_rules,
            "final_rules": self.clf.model.get_rules_size(),
        }
