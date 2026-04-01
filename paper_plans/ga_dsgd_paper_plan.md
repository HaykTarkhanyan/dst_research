# Evolutionary Rule Optimization for Dempster-Shafer Classifiers with Multi-Objective Fitness

## Paper Plan

---

## Motivation

DSGD++ involves two fundamentally different optimization problems:

| Problem | Nature | Current Approach | GA Fit |
|---------|--------|-----------------|--------|
| **MAF optimization** | Continuous (probability simplex) | Gradient descent | Moderate — GD is already good here |
| **Rule generation/selection** | Discrete/combinatorial | Fixed single-feature thresholds | **Excellent** — this is where GAs shine |
| **Hyperparameter tuning** | Mixed discrete/continuous | Manual | Good |
| **Frame of discernment design** | Combinatorial (which subsets?) | Only singletons + full set | **Excellent** |

**Key insight:** Gradient descent handles continuous MAFs well, but the discrete decisions (which rules? which focal elements? which combination of hyperparameters?) are exactly what GAs were designed for.

---

## Core Ideas

### Idea 1: Evolutionary Rule Generation for DSGD (Primary Contribution)

Currently rules are either single-feature thresholds or mined by RIPPER/SkopeRules. A GA can **evolve** rules directly, using DST-specific fitness.

**Chromosome encoding:**

```
Each chromosome = one rule

Gene 1: feature_index       (integer, 0 to num_features-1)
Gene 2: operator            (categorical: <, >, <=, >=)
Gene 3: threshold           (float, from feature range)
Gene 4: second_feature      (integer or None — for conjunctive rules)
Gene 5: second_operator     (categorical)
Gene 6: second_threshold    (float)
```

A chromosome `[2, >, 5.0, 7, <=, 3.0]` encodes:
*"if feature_2 > 5.0 AND feature_7 <= 3.0"*

**Fitness function — the key novelty:**

```
fitness(rule) = alpha * confidence(rule)           # from DSGD++ confidence
              + beta  * coverage(rule)             # fraction of data covered
              - gamma * overlap(rule, existing)    # penalize redundant rules
```

Where `confidence(rule)` is exactly the DSGD++ confidence based on representativeness. The GA's fitness function is informed by DST — the GA isn't just finding accurate rules, it's finding rules that will have strong, low-uncertainty MAFs.

**Why this is better than RIPPER/SkopeRules for DSGD:**
- RIPPER optimizes for accuracy. The GA optimizes for *DST confidence* — a different and more appropriate objective for a DST classifier.
- RIPPER generates rules greedily (one at a time). GA explores the space of rule *sets* holistically.
- The fitness function can directly incorporate uncertainty reduction as an objective.

---

### Idea 2: Multi-Objective GA for the Accuracy-Interpretability-Uncertainty Pareto Frontier

DSGD++ inherently involves competing objectives. A multi-objective GA (NSGA-II or NSGA-III) can find the Pareto frontier.

**Chromosome:** A binary vector of length N (total candidate rules), where 1 = include rule, 0 = exclude.

**Three objectives (all to be optimized):**

```
Objective 1: Maximize accuracy (train DSGD++ with selected rules, evaluate)
Objective 2: Minimize rule count (sum of chromosome = number of active rules)
Objective 3: Minimize average uncertainty (mean mass of full set across rules)
```

**Output:** A Pareto frontier showing the tradeoff:

```
Accuracy
  |          * (many rules, high acc, low uncertainty)
  |       *
  |     *     *
  |   *          *  (few rules, interpretable, higher uncertainty)
  |  *
  +-------------------> Number of Rules
```

The researcher picks their preferred operating point. This is a publishable figure.

**Implementation:** pymoo has NSGA-II, NSGA-III, and visualization built in.

---

### Idea 3: Hybrid GA-GD (Evolutionary MAF Initialization)

Instead of clustering-based initialization OR random initialization, use a GA to evolve good MAF starting points, then refine with gradient descent.

```
HybridGA-GD:

1. Generate population of P random MAF initializations
2. For each individual:
   - Run GD for a FEW epochs (e.g., 10, not full convergence)
   - Evaluate fitness = -loss after 10 epochs
3. Selection, crossover, mutation on MAF vectors
4. Repeat for G generations
5. Take the best individual, run full GD to convergence
```

**Why interesting:** This is a meta-optimization — the GA finds the basin of attraction that leads to the best final solution after GD. It's like the clustering-based init but lets the data speak for itself without assuming cluster structure.

**Comparison point:** Three initialization strategies:
- Random (Peñafiel 2020)
- Clustering-based (DSGD++ 2025)
- GA-evolved (new)

If GA-evolved init beats clustering-based, that's a strong result. If clustering-based is competitive with GA (which is much more expensive), that validates the approach from a different angle.

---

### Idea 4: Evolving the Frame of Discernment

Currently DSGD++ only assigns mass to singleton classes + the full set (total uncertainty). But DST allows mass on ANY subset. The problem: 2^n subsets is too many.

A GA can search for the **optimal sparse frame** — which non-singleton subsets to include.

**Chromosome:** For n classes, each possible non-singleton subset is a gene (1 = include in frame, 0 = exclude).

For 4 classes {A, B, C, D}, candidate subsets are:
{A,B}, {A,C}, {A,D}, {B,C}, {B,D}, {C,D}, {A,B,C}, {A,B,D}, {A,C,D}, {B,C,D}, {A,B,C,D}

Chromosome `[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]` means:
Use singletons + {A,B} + {B,C} + {A,B,C,D}

**Fitness:** Accuracy of DSGD++ trained with this frame - penalty for frame size.

**Novelty:** Nobody has used GAs to design the frame of discernment. This directly addresses the O(2^n) scalability problem.

---

### Idea 5: Coevolutionary Rules + MAFs

The most ambitious idea — coevolve two populations simultaneously:

- **Population 1:** Rule sets (discrete — which rules to use)
- **Population 2:** MAF initializations (continuous — starting mass values)

Rules and MAFs coevolve: good rules help MAFs converge faster, good MAFs make rules more effective. This is a cooperative coevolutionary algorithm.

---

## Proposed Paper Structure

Combine Ideas 1 + 2 as the main paper.

### Contributions

1. **GA-based rule generation** with DST-specific fitness (confidence + coverage - overlap)
2. **Multi-objective Pareto frontier** (accuracy vs. interpretability vs. uncertainty) via NSGA-II
3. **Comparison:** GA-evolved rules vs. single-feature vs. RIPPER vs. SkopeRules within DSGD++
4. **Hybrid architecture:** GA handles discrete rule selection, GD handles continuous MAF optimization — each algorithm does what it's best at

---

## Experimental Design

| Experiment | What it shows |
|-----------|--------------|
| **E1: GA-evolved rules vs. baselines** | Compare GA rules vs. single-feature vs. RIPPER vs. SkopeRules in DSGD. Metrics: accuracy, F1, uncertainty, rule count, training time. |
| **E2: Pareto frontier** | 3-objective NSGA-II. Plot the accuracy-interpretability-uncertainty tradeoff surface. |
| **E3: Fitness function ablation** | Compare fitness = accuracy only, vs. confidence only, vs. full DST-aware fitness. Show DST-aware fitness produces better DSGD++ models. |
| **E4: GA convergence analysis** | Generations vs. fitness. Show GA converges in reasonable time. |
| **E5: Evolved rule analysis** | Inspect what rules the GA discovers. Are they more complex? More interpretable? Do they capture feature interactions that single-feature rules miss? |
| **E6: GA-init vs. clustering-init for MAFs** (Idea 3) | Side experiment: does GA-evolved MAF init beat or match clustering-based? Validates DSGD++ from a different angle. |

---

## Implementation Sketch

```python
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
import numpy as np

class DSGDRuleOptimization(Problem):
    def __init__(self, X, y, candidate_rules, dsgd_model):
        # Binary decision: include/exclude each candidate rule
        super().__init__(
            n_var=len(candidate_rules),
            n_obj=3,    # accuracy, -rule_count, -uncertainty
            n_constr=1, # at least 3 rules
            xl=0, xu=1,
            type_var=int
        )
        self.X, self.y = X, y
        self.candidate_rules = candidate_rules
        self.dsgd = dsgd_model

    def _evaluate(self, X_pop, out, *args, **kwargs):
        f1, f2, f3 = [], [], []
        for individual in X_pop:
            selected_rules = [r for r, active in
                            zip(self.candidate_rules, individual) if active]
            # Train DSGD++ with selected rules
            self.dsgd.set_rules(selected_rules)
            self.dsgd.fit(self.X, self.y)

            f1.append(-self.dsgd.accuracy())      # maximize accuracy
            f2.append(len(selected_rules))          # minimize rules
            f3.append(self.dsgd.mean_uncertainty()) # minimize uncertainty

        out["F"] = np.column_stack([f1, f2, f3])
        out["G"] = np.array([3 - sum(ind) for ind in X_pop])  # >= 3 rules
```

---

## Assessment

| Aspect | Rating |
|--------|--------|
| **Novelty** | High — GA with DST-specific fitness for rule evolution is new |
| **Feasibility** | High — pymoo handles the GA; DSGD++ handles MAF optimization |
| **Implementation** | ~2-3 weeks |
| **Risk** | Medium — GA may be slow if DSGD++ training is in the inner loop. Mitigation: use short GD runs (10 epochs) for fitness evaluation, full training only for final solution. |
| **Best venues** | Information Fusion, Applied Soft Computing, Evolutionary Computation, GECCO |
| **Connection to other plans** | Can combine with rule induction plan — GA generates candidate rules, uncertainty-guided loop refines them |

---

## Connection to Other Paper Plans

```
         Rule Induction Plan
         (RIPPER/SkopeRules)
                |
                v
    GA Plan (evolve rules with DST fitness)
                |                    \
                v                     v
    Federated Plan              Multi-Objective
    (combine across clients)    Pareto Frontier
```

The GA plan can be either:
- A **standalone paper** (GA rule evolution + NSGA-II Pareto frontier)
- A **section within the rule induction paper** (GA as another rule source alongside RIPPER/SkopeRules)
- A **bridge to the federated paper** (GA evolves rules locally at each client, then federate MAFs)
