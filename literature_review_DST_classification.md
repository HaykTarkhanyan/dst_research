# Comprehensive Literature Review: Dempster-Shafer Theory Applied to Classification and Machine Learning

**Prepared for:** Aik Tarkhanyan & Ashot Harutyunyan (DSGD++ authors)
**Date:** March 31, 2026
**Scope:** 2018-2026, with emphasis on 2020-2026

---

## Table of Contents

1. [Recent DST-Based Classifiers (2020-2026)](#1-recent-dst-based-classifiers-2020-2026)
2. [Mass Function Initialization and Learning](#2-mass-function-initialization-and-learning)
3. [Uncertainty Quantification in Evidence Theory](#3-uncertainty-quantification-in-evidence-theory)
4. [Interpretable/Explainable ML with DST](#4-interpretableexplainable-ml-with-dst)
5. [Alternatives to Dempster's Rule of Combination](#5-alternatives-to-dempsters-rule-of-combination)

---

## 1. Recent DST-Based Classifiers (2020-2026)

### 1.1 Foundational DSGD Work

**[P1] Penafiel, S., Baloian, N., Sanson, H., Pino, J.A. "Applying Dempster-Shafer Theory for Developing a Flexible, Accurate and Interpretable Classifier." Expert Systems with Applications, 148, 113262 (2020).**
- **Venue:** Expert Systems with Applications (Elsevier)
- **Summary:** Introduces the DSGD classifier, which combines Dempster-Shafer theory with gradient descent optimization to learn mass assignment functions (MAFs). The model is rule-based, inherently interpretable, can handle missing information, and achieves accuracy comparable to SVM and Random Forest. Each rule's MAF is optimized via gradient descent from the training data. This is the foundational paper for the DSGD line of research that DSGD++ builds upon.
- **Link:** https://www.sciencedirect.com/science/article/abs/pii/S0957417420300877

**[P2] Penafiel, S. "Interpretable Method for General Classification Using Dempster-Shafer Theory." Master's Thesis, Universidad de Chile (2021).**
- **Venue:** Universidad de Chile Repository
- **Summary:** Penafiel's thesis generalizes the DSGD framework to multi-class problems, presenting a model as interpretable as Decision Trees with precision comparable to traditional ML methods. It uses gradient descent to optimize mass values while preserving interpretability. The thesis also applies the classifier to stroke risk detection using clinical data.
- **Link:** https://repositorio.uchile.cl/bitstream/handle/2250/177533/Interpretable-method-for-general-classification-using-Dempster-Shafer-theory.pdf

**[P3] Penafiel, S., Baloian, N., Sanson, H., Pino, J.A. "Predicting Stroke Risk With an Interpretable Classifier." IEEE Access, 9 (2021).**
- **Venue:** IEEE Access
- **Summary:** Applies the DSGD classifier to predicting stroke risk using health checkup data from a hospital in Okayama, Japan. Demonstrates the practical value of DST-based interpretable classification in healthcare, where model transparency is critical. Achieves competitive accuracy while providing transparent, rule-based explanations.
- **Link:** https://ieeexplore.ieee.org/document/9306826/

**[P4] Tarkhanyan, A., Harutyunyan, A. "Improving the DSGD Classifier with an Initialization Technique for Mass Assignment Functions." CODASSCA 2024, Logos, 137-142 (2024).**
- **Venue:** CODASSCA Workshop
- **Summary:** Preliminary version of the DSGD++ work. Introduces the concept of using arithmetic mean of data points as the most representative point for MAF initialization instead of random assignment. Later improved in DSGD++ by using KMeans clustering.

**[P5] Tarkhanyan, A., Harutyunyan, A. "DSGD++: Reducing Uncertainty and Training Time in the DSGD Classifier through a Mass Assignment Function Initialization Technique." Journal of Universal Computer Science, 31(9), 1004-1014 (2025).**
- **Venue:** J.UCS
- **Summary:** The user's paper. Introduces clustering-based confidence scores for MAF initialization, achieving 1.6x training speedup and 2.1x uncertainty reduction while preserving accuracy. Proposes a new uncertainty measure based on the harmonic mean of (1-uncertainty) and the mass ratio between the most and second-most probable classes.
- **Link:** https://lib.jucs.org/article/164745/

### 1.2 Extensions of the DSGD Framework

**[P6] Baloian, N., Davtyan, E., Petrosyan, K., Poghosyan, A., Harutyunyan, A., Penafiel, S. "Embedded Interpretable Regression using Dempster-Shafer Theory." Proceedings of the 4th CODASSCA Workshop (2024).**
- **Venue:** CODASSCA 2024
- **Summary:** Extends the DSGD framework from classification to regression (EVREG). Uses an interpretable regression method based on Dempster-Shafer theory with a discount function based on the contribution of each dimension. Maintains the interpretability strengths of the classification approach while handling continuous target variables.

**[P7] Valdivia, R., Baloian, N., Chahverdian, M., Adamyan, A., Harutyunyan, A. "An Explainable Clustering Algorithm using Dempster-Shafer Theory." Proceedings of the 4th CODASSCA Workshop (2024).**
- **Venue:** CODASSCA 2024
- **Summary:** Extends DST to clustering problems. Combines multiple sources of evidence using Dempster's rule to aggregate evidence into a joint belief function for cluster assignment. Each rule is transparent and allows domain experts to verify or modify cluster interpretations.
- **Link:** https://lib.jucs.org/article/164694/

**[P8] Baloyan, A., Aramyan, A., Baloian, N., Poghosyan, A., Harutyunyan, A., Penafiel, S. "An Empirical Analysis of Feature Engineering for Dempster-Shafer Classifier as a Rule Validator." Proceedings of the 4th CODASSCA Workshop (2024).**
- **Venue:** CODASSCA 2024
- **Summary:** Demonstrates the DSGD algorithm's robustness to class imbalance and its ability to capture various numeric interactions among features. Shows that the DST-based classifier can serve as a rule validator, verifying hypotheses encoded as rules about data patterns.

### 1.3 Evidential Neural Networks and Deep Learning

**[P9] Sensoy, M., Kaplan, L., Kandemir, M. "Evidential Deep Learning to Quantify Classification Uncertainty." NeurIPS 2018, 31 (2018).**
- **Venue:** Advances in Neural Information Processing Systems (NeurIPS)
- **Summary:** Landmark paper that places a Dirichlet distribution on class probabilities, treating neural network predictions as subjective opinions. Learns the evidence function via a deterministic neural net. Achieves strong performance on OOD detection and adversarial robustness. Has become the foundational reference for the "evidential deep learning" (EDL) paradigm. Highly relevant as an alternative approach to incorporating DST into deep learning.
- **Link:** https://arxiv.org/abs/1806.01768

**[P10] Denoeux, T. "Logistic Regression, Neural Networks and Dempster-Shafer Theory: A New Perspective." Knowledge-Based Systems, 176, 54-67 (2019).**
- **Venue:** Knowledge-Based Systems (Elsevier)
- **Summary:** Shows that logistic regression and multilayer neural networks can be viewed as converting features into DS mass functions and combining them by Dempster's rule. The mass function is more informative than probability output, distinguishing between lack of evidence and conflicting evidence. Provides theoretical grounding for interpreting hidden unit outputs through the lens of belief functions.
- **Link:** https://www.sciencedirect.com/science/article/abs/pii/S0950705119301558

**[P11] Tong, Z., Xu, P., Denoeux, T. "An Evidential Classifier Based on Dempster-Shafer Theory and Deep Learning." Neurocomputing, 450, 275-293 (2021).**
- **Venue:** Neurocomputing (Elsevier)
- **Summary:** Proposes a deep classifier combining CNN with a DS layer. Convolutional layers extract features, which are converted into mass functions and aggregated by Dempster's rule. An expected utility layer performs set-valued classification. Demonstrates improved classification on image recognition, signal processing, and semantic-relationship tasks, allowing cautious decisions by assigning confusing patterns to multi-class sets.
- **Link:** https://www.sciencedirect.com/science/article/abs/pii/S0925231221004525

**[P12] Denoeux, T. "An Evidential Neural Network Model for Regression Based on Random Fuzzy Numbers." BELIEF 2022, Springer LNAI 13506, 57-66 (2022).**
- **Venue:** International Conference on Belief Functions (BELIEF)
- **Summary:** Introduces a distance-based neural network for regression where uncertainty is quantified by belief functions. Interprets distances to prototypes as evidence represented by Gaussian Random Fuzzy Numbers (GRFNs), combined by a generalized product-intersection rule. Network output is a GRFN summarized by three numbers: most plausible value, variability, and epistemic uncertainty.
- **Link:** https://arxiv.org/abs/2208.00647

**[P13] Denoeux, T. "Uncertainty Quantification in Regression Neural Networks Using Likelihood-Based Belief Functions." BELIEF 2024 (2024).**
- **Venue:** BELIEF 2024
- **Summary:** Introduces Epistemic Random Fuzzy Set (ERFS) theory, a new model of uncertainty based on random fuzzy sets with a new combination rule generalizing Dempster's rule. Provides a framework for reasoning with fuzzy and uncertain evidence, advancing the theoretical foundations of evidential neural networks for regression.

**[P14] Pichon, F., Diene, S.M., Denoeux, T., Ramel, S., Mercier, D. "r-ERBFN: An Extension of the Evidential RBFN Accounting for the Dependence Between Positive and Negative Evidence." SUM 2024, Springer LNAI (2024).**
- **Venue:** Scalable Uncertainty Management (SUM) Conference
- **Summary:** Shows that an RBFN with softmax output amounts to combining positive and negative evidence for each class via Dempster's rule. Reveals a set of latent belief functions obtained by considering all possible dependence structures between positive and negative evidence. Performance is enhanced by optimizing these correlation parameters.
- **Link:** https://link.springer.com/chapter/10.1007/978-3-031-76235-2_26

**[P15] Gong, C. et al. "An Evidential Deep Network Based on Dempster-Shafer Theory for Large Dataset." HAL preprint, hal-04448387 (2024).**
- **Venue:** HAL (Normandie Universite)
- **Summary:** Proposes a novel DST-based deep architecture capable of handling large-scale datasets like ImageNet (1000 classes). Multiple expert CNNs predict mass functions, which are merged via Dempster's rule. Includes an efficient algorithm for identifying the best class set among 2^K possibilities with O(K log K) complexity. Achieves 77.65% accuracy on ImageNet (vs. 77.77% for the probabilistic model). Also applied to OOD detection.
- **Link:** https://normandie-univ.hal.science/hal-04448387v1

### 1.4 Evidential k-NN and Random Forests

**[P16] Su, Z., Hu, Q., Denoeux, T. "A Distributed Rough Evidential K-NN Classifier: Integrating Feature Reduction and Classification." IEEE Transactions on Fuzzy Systems, 29(8), 2322-2335 (2021).**
- **Venue:** IEEE Transactions on Fuzzy Systems
- **Summary:** Addresses the curse of dimensionality in the Evidential K-NN (EK-NN) classifier by proposing a rough evidential K-NN (REK-NN) that integrates rough set-based feature selection with evidential classification. Unlike stepwise methods, feature selection and learning are performed simultaneously by minimizing a neighborhood pignistic decision error rate.
- **Link:** https://ieeexplore.ieee.org/document/9103960/

**[P17] Gong, C., Su, Z., Wang, P., Wang, Q. "Belief Entropy Tree and Random Forest: Learning from Data with Continuous Attributes and Evidential Labels." Entropy, 24(5), 605 (2022).**
- **Venue:** Entropy (MDPI)
- **Summary:** Proposes a decision tree that uses belief entropy (based on BBA) as the attribute selection criterion. Data is fitted to Gaussian Mixture Models per attribute, from which BBAs and belief entropy are calculated. The tree handles continuous attributes without discretization and is extended to a belief function-based random forest that can output either precise or mass-valued predictions.
- **Link:** https://www.mdpi.com/1099-4300/24/5/605

**[P18] Hoarau, A., Martin, A., Dubois, J.-C., Le Hegarat-Mascle, S. "Evidential Random Forests." Expert Systems with Applications, 230, 120614 (2023).**
- **Venue:** Expert Systems with Applications
- **Summary:** Proposes Evidential Decision Trees for rich (uncertain/imprecise) labels that are robust to overfitting using a conflict-based splitting criterion. Extends to Evidential Random Forests that handle imperfectly labeled data from multiple contributors. The models can predict rich labels useful for active learning. Directly relevant to the DSGD++ work as both address uncertainty in rule/tree-based classifiers.
- **Link:** https://www.sciencedirect.com/science/article/abs/pii/S0957417423011545

### 1.5 DST for Feature/Classifier Fusion

**[P19] Kurniawan, H., Mambo, M. "Feature Fusion for Improved Classification: Combining Dempster-Shafer Theory and Multiple CNN Architectures." LNCS 14879, Springer (2024).**
- **Venue:** International Conference on Computational Science (ICCS) 2024 / Springer LNCS
- **Summary:** Proposes a DST-based algorithm to fuse features from multiple pre-trained CNNs (VGG, ResNet, DenseNet). Features from final convolutional layers are converted to mass functions and combined via DST. Achieves 5.4% and 8.4% improvement on CIFAR-10 and CIFAR-100 over the best individual models. Demonstrates DST as a robust framework for managing uncertainty across diverse deep learning models.
- **Link:** https://arxiv.org/abs/2405.20230

**[P20] Yaghoubi, V., Cheng, L., Van Paepegem, W., Kersemans, M. "A Novel Multi-Classifier Information Fusion Based on Dempster-Shafer Theory." Structural Health Monitoring, 21(5) (2022).**
- **Venue:** Structural Health Monitoring (SAGE)
- **Summary:** Proposes a multi-classifier fusion framework using DST for vibration-based fault detection. Introduces preprocessing metrics to measure and mitigate conflict between classifier evidence before applying Dempster's combination rule. Demonstrates practical engineering applications of DST-based classifier fusion.
- **Link:** https://journals.sagepub.com/doi/abs/10.1177/14759217211007130

### 1.6 Surveys and Review Papers

**[P21] Zhao, K. et al. "A Survey: Optimization and Applications of Evidence Fusion Algorithm Based on Dempster-Shafer Theory." Applied Soft Computing, 124, 109075 (2022).**
- **Venue:** Applied Soft Computing (Elsevier)
- **Summary:** Comprehensive survey covering DST evidence fusion optimization across multiple domains (risk analysis, fault diagnosis, sensor networks, image processing, target tracking). Reviews methods for modifying combination rules and preprocessing evidence to handle counter-intuitive results from conflicting evidence. Essential reference for understanding the landscape of DST optimization research.
- **Link:** https://www.sciencedirect.com/science/article/abs/pii/S1568494622003696

**[P22] Chen, Z. et al. "A Comprehensive Survey on Evidential Deep Learning and Its Applications." arXiv:2409.04720 (2024).**
- **Venue:** arXiv preprint (September 2024)
- **Summary:** The most recent comprehensive survey on Evidential Deep Learning (EDL). Covers the theoretical foundation through subjective logic theory and presents advances from four perspectives: evidence collection reformulation, uncertainty estimation via OOD samples, training strategies, and evidential regression. Catalogs applications in medical imaging, autonomous driving, action recognition, and molecular discovery.
- **Link:** https://arxiv.org/abs/2409.04720

### 1.7 Miscellaneous DST-Based Classifiers

**[P23] Li, C. et al. "A New Correlation Belief Function in Dempster-Shafer Evidence Theory and Its Application in Classification." Scientific Reports, 13, 7765 (2023).**
- **Venue:** Scientific Reports (Nature)
- **Summary:** Proposes a new correlation belief function that transfers belief from a proposition to related propositions to avoid information loss during fusion. Addresses conflict management in D-S theory. Experiments on UCI datasets show the method assigns higher belief to correct propositions and better expresses conflict among data than existing approaches.
- **Link:** https://www.nature.com/articles/s41598-023-34577-y

**[P24] Sedlacek, O., Bartos, V. "Fusing Heterogeneous Data for Network Asset Classification -- A Two-layer Approach." IEEE NOMS 2024 (2024).**
- **Venue:** IEEE Network Operations and Management Symposium (NOMS)
- **Summary:** Investigates using the statistical distribution of classes in allocating mass values for DST-based network asset classification. Directly relevant to DSGD++ as it explores an alternative to random MAF initialization. Referenced in the DSGD++ paper as a related approach to initializing mass functions.

**[P25] Shiraishi, T. et al. "A Class Inference Scheme With Dempster-Shafer Theory for Learning Fuzzy-Classifier Systems." ACM Transactions on Evolutionary Learning and Optimization (2025).**
- **Venue:** ACM TELO
- **Summary:** Proposes a novel class inference scheme for Learning Fuzzy-Classifier Systems (LFCSs) based on DST. Calculates belief masses for each class and an "I don't know" state from fuzzy rules. Unlike conventional voting-based approaches, the scheme considers uncertainty explicitly. Shows statistically significant improvements on 30 real-world datasets.
- **Link:** https://dl.acm.org/doi/10.1145/3717613

**[P26] Mehravaran, Z. et al. "Dempster-Shafer Deep Capsule Attention Model (DDCAM)." Multimedia Tools and Applications (2025).**
- **Venue:** Multimedia Tools and Applications (Springer)
- **Summary:** Combines 3D convolution, capsule networks, and attention mechanisms with DST for uncertainty management and feature selection. DDCAM outperforms state-of-the-art capsule networks on MNIST, CIFAR10, SVHN, and Fashion MNIST, improving accuracy by 1.29% on CIFAR10 and 0.76% on SVHN.
- **Link:** https://link.springer.com/content/pdf/10.1007/s11042-025-20866-3.pdf

**[P27] Yuan, B., Yue, X., Lv, Y., Denoeux, T. "Evidential Deep Neural Networks for Uncertain Data Classification." KSEM 2020, Springer LNAI (2020).**
- **Venue:** KSEM 2020
- **Summary:** Applies evidential approaches specifically to uncertain data classification, where the input data itself is imprecise or incomplete. Extends the evidential neural network framework to handle epistemic uncertainty in both inputs and outputs.

---

## 2. Mass Function Initialization and Learning

This section is most directly relevant to DSGD++, which proposes clustering-based MAF initialization.

**[P1] Penafiel et al. (2020) -- see above.** Uses random MAF initialization: empty set mass = 0, complete set (uncertainty) = 0.8, remaining 0.2 randomly distributed among singleton classes. DSGD++ improves upon this.

**[P24] Sedlacek & Bartos (2024) -- see above.** Uses statistical class distribution for mass allocation, an intermediate step between random and clustering-based initialization.

**[P28] Li, D. et al. "A New Basic Probability Assignment Generation and Combination Method for Conflict Data Fusion in the Evidence Theory." Scientific Reports, 13, 8672 (2023).**
- **Venue:** Scientific Reports (Nature)
- **Summary:** Proposes BPA generation based on cosine similarity and belief entropy, using Mahalanobis distance to measure similarity. Addresses how BPA determination remains an open issue in evidence theory. Uses cosine similarity and belief entropy to measure reliability and uncertainty of generated BPAs.
- **Link:** https://www.nature.com/articles/s41598-023-35195-4

**[P29] E-RBFN (Evidential Radial Basis Function Network) approach. "Basic Belief Assignment Determination Based on Radial Basis Function Network." (2024).**
- **Venue:** Conference on Information and Knowledge Computing
- **Summary:** Constructs end-to-end Evidential RBF Networks where the input is a sample and the output is the corresponding BBA/mass function. Can directly determine mass values for all focal elements without manual initialization. Represents a data-driven approach to mass function generation.

**[P30] McGuire, T. et al. "A Dempster-Shafer, Fusion-Based Approach for Malware Detection." Mathematics, 13(16), 2677 (2025).**
- **Venue:** Mathematics (MDPI)
- **Summary:** Systematically compares three approaches for converting classifier outputs into mass functions: (1) global confidence using fixed performance metrics, (2) class-specific confidence with per-class values, and (3) computationally optimized confidence values. Finds that all approaches yield comparable performance, though fixed values offer computational and interpretability advantages. Directly relevant to understanding how mass function construction choices affect classification outcomes.
- **Link:** https://www.mdpi.com/2227-7390/13/16/2677

**[P10] Denoeux (2019) -- see above.** Shows how logistic regression inherently performs mass function construction and combination, providing a theoretical perspective on what constitutes an "optimal" mass function for classification.

**[P5] DSGD++ (Tarkhanyan & Harutyunyan, 2025) -- the user's paper.** Proposes the main innovation: (1) computing "representativeness" of data points via KMeans/DBSCAN clustering, (2) aggregating representativeness into rule "confidence," and (3) using confidence to initialize MAFs with the modal class receiving mass = confidence, remainder distributed equally. This replaces random initialization with an informed starting point.

---

## 3. Uncertainty Quantification in Evidence Theory

### 3.1 Traditional and New Uncertainty Measures

**[P31] Jirousek, R., Shenoy, P. "A Numerical Comparative Study of Uncertainty Measures in the Dempster-Shafer Evidence Theory." Information Sciences, 639, 118929 (2023).**
- **Venue:** Information Sciences (Elsevier)
- **Summary:** Compares a wide range of uncertainty measures proposed for DST through numerical simulations. Analyzes similarities and differences using rank correlation, hierarchical clustering, and centrality analysis. Key finding: measures with similar formulations do not necessarily have similar numerical properties. Essential reference for DSGD++'s proposed harmonic-mean uncertainty measure.
- **Link:** https://www.sciencedirect.com/science/article/pii/S0020025523006126

**[P32] Deng, Y. "Deng Entropy." Chaos, Solitons & Fractals, 91, 549-553 (2016; widely cited in 2020-2024).**
- **Venue:** Chaos, Solitons & Fractals (Elsevier)
- **Summary:** Proposes Deng entropy as a generalization of Shannon entropy to belief functions. When the BPA defines a probability measure, Deng entropy equals Shannon entropy. Has spawned a large body of follow-up research on improved belief entropy measures. The baseline against which many subsequent uncertainty measures are compared.

**[P33] Qin, M., Tang, Y., Wen, J. "An Improved Total Uncertainty Measure in the Evidence Theory and Its Application in Decision Making." Entropy, 22(4), 487 (2020).**
- **Venue:** Entropy (MDPI)
- **Summary:** Proposes a two-component uncertainty measure: (1) improved Dubois-Prade entropy for non-specificity, considering element count in FOD, and (2) Nguyen entropy for conflict. Can differentiate BPAs with equal belief values on different FODs, unlike Deng entropy. Directly relevant to DSGD++'s approach of combining multiple aspects of uncertainty.
- **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC7516972/

**[P34] Gao, X., Deng, Y. "The Pseudo-Pascal Triangle of Maximum Deng Entropy." International Journal of Computers, Communications & Control, 15(1) (2020).**
- **Venue:** Int. J. Computers, Communications & Control
- **Summary:** Investigates maximum Deng entropy properties, extending the understanding of what constitutes maximum uncertainty in evidence theory.

**[P35] Pan, Q. et al. "A New Belief Entropy in Dempster-Shafer Theory Based on Basic Probability Assignment and the Frame of Discernment." Entropy, 22(9), 926 (2020).**
- **Venue:** Entropy (MDPI)
- **Summary:** Proposes an entropy measure that explicitly considers the frame of discernment structure. Takes both belief function and plausibility function as bounds, using the center and span of the belief interval to define total uncertainty.
- **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC7517227/

**[P36] Wang, D. et al. "A New Belief Interval-Based Total Uncertainty Measure for Dempster-Shafer Theory." Information Sciences, 642, 119152 (2023).**
- **Venue:** Information Sciences (Elsevier)
- **Summary:** Proposes an uncertainty measure based on belief intervals that satisfies non-negativity, generalized set consistency, coherent range, probability consistency, monotonicity, invariance, and maximum entropy properties. Directly relevant to DSGD++ because the user's harmonic-mean approach also moves beyond the simple "mass of complete set" measure.
- **Link:** https://www.sciencedirect.com/science/article/abs/pii/S0020025523007351

**[P37] An, J., Deng, Y. "A Dual Measure of Uncertainty: The Deng Extropy." Entropy, 22(5), 582 (2020).**
- **Venue:** Entropy (MDPI)
- **Summary:** Defines the Deng extropy as the dual of Deng entropy and studies their interrelationship. Provides a complementary uncertainty perspective to entropy-based measures.
- **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC7517106/

**[P38] Huang, Z. et al. "Evaluation of Belief Entropies: From the Perspective of Evidential Neural Network." Artificial Intelligence Review, 58, 130 (2025).**
- **Venue:** Artificial Intelligence Review (Springer)
- **Summary:** Evaluates various belief entropy measures from the perspective of their utility within evidential neural networks. Bridges the gap between theoretical uncertainty measures and practical deep learning applications. Provides empirical guidance on which entropy measures are most useful in practice.
- **Link:** https://link.springer.com/article/10.1007/s10462-025-11130-z

### 3.2 Uncertainty Decomposition (Aleatoric vs. Epistemic)

**[P39] Ulmer, D., Hardmeier, C., Frellsen, J. "Reliable Classifications with Guaranteed Confidence Using the Dempster-Shafer Theory of Evidence." ECML PKDD 2024 (2024).**
- **Venue:** ECML PKDD 2024 / Springer LNCS
- **Summary:** Trains a neural network to return DS mass functions, combined with conformal prediction for guaranteed confidence. Yields smaller, more informative prediction sets at bounded error levels, especially for high epistemic uncertainty samples. Also proposes a reduced-complexity variant for tasks with 40+ classes. Demonstrates how DST naturally decomposes uncertainty into aleatoric and epistemic components.
- **Link:** https://link.springer.com/chapter/10.1007/978-3-031-70344-7_6

**[P40] Shen, M., Ryu, J.J., Ghosh, S. et al. "Are Uncertainty Quantification Capabilities of Evidential Deep Learning a Mirage?" NeurIPS 2024 (2024).**
- **Venue:** NeurIPS 2024
- **Summary:** Critical examination of EDL's uncertainty quantification. Finds that EDL's learned epistemic uncertainties can be unreliable (non-vanishing even with infinite data). Argues EDL methods are better interpreted as OOD detection via energy-based models. Proposes incorporating explicit model uncertainty to improve faithful uncertainty quantification. Important counterpoint to the EDL paradigm.
- **Link:** https://proceedings.neurips.cc/paper_files/paper/2024/hash/c3177be226ee12e34d6ba3b5e6fe6a5b-Abstract-Conference.html

**[P41] Kohl, P. et al. "On Several New Dempster-Shafer-Inspired Uncertainty Measures Applicable for Active Learning." ECML PKDD 2023 Workshop (2023).**
- **Venue:** ECML PKDD Workshop / Springer LNCS
- **Summary:** Proposes a new family of uncertainty functions inspired by DST for active learning uncertainty sampling. The key insight is that uncertainty functions based on belief functions can capture informativeness of unlabeled samples better than standard entropy or margin-based approaches. Relevant to DSGD++ because both works propose novel uncertainty measures within the DST framework.
- **Link:** https://link.springer.com/chapter/10.1007/978-3-031-50959-9_33

### 3.3 Relevance to DSGD++ Uncertainty Measure

The DSGD++ paper proposes a harmonic mean of (1-uncertainty) and the normalized mass ratio as a new rule importance measure. This approach is most closely related to:
- **[P33]** (two-component measures combining non-specificity and conflict)
- **[P36]** (belief interval-based measures going beyond single-valued uncertainty)
- **[P31]** (numerical comparison showing that similar formulations don't guarantee similar behavior)
- **[P38]** (practical evaluation of entropy measures in neural network contexts)

The key novelty in DSGD++ is that it explicitly accounts for how well a rule discriminates between classes (via mass ratio), not just the overall uncertainty level. This is conceptually similar to how [P33] combines non-specificity with conflict, but applied at the individual rule level rather than globally.

---

## 4. Interpretable/Explainable ML with DST

**[P1] Penafiel et al. (2020) -- see above.** The foundational work establishing DST-based classifiers as inherently interpretable: every rule is simple, understandable, and applies evenly to all data points.

**[P3] Penafiel et al. (2021) -- see above.** Demonstrates interpretable stroke risk prediction where model transparency is critical for clinical acceptance.

**[P8] Baloyan et al. (2024) -- see above.** Shows DST classifier as a "rule validator" for feature engineering, demonstrating how interpretability enables domain expert interaction.

**[P25] Shiraishi et al. (2025) -- see above.** Enhances transparency in fuzzy classifier systems by explicitly modeling an "I don't know" state via belief masses. Shows interpretability gains on 30 datasets.

**[P42] Hajiabadi, H. et al. "A Study on Automated Problem Troubleshooting in Cloud Environments with Rule Induction and Verification." Applied Sciences, 14(3), 1047 (2024).**
- **Venue:** Applied Sciences (MDPI)
- **Summary:** Applies DST-based rule verification as a "what-if" analysis tool in cloud troubleshooting. Uses an interpretable classifier with expert-encoded rule conditions on system features. Demonstrates how DST's belief function framework enables transparent, verifiable decision-making in operational settings.
- **Link:** https://www.mdpi.com/2076-3417/14/3/1047

**[P43] Li, Y. et al. "Interpretable Systems Based on Evidential Prospect Theory for Decision-Making." Applied Intelligence, 52, 3806-3826 (2022).**
- **Venue:** Applied Intelligence (Springer)
- **Summary:** Combines evidential theory (Dempster-Shafer) with prospect theory for interpretable decision-making systems. Addresses how to resolve basic probability computation and conflicting evidence combination using triangular fuzzy numbers and multi-criteria decision-making methods, offering an alternative to standard Dempster combination.
- **Link:** https://link.springer.com/article/10.1007/s10489-022-03276-y

**[P44] Xu, P. et al. "Dynamic Feature Selection Based on Rule-Based Learning for Explainable Classification with Uncertainty Quantification." arXiv:2508.02566 (2025).**
- **Venue:** arXiv preprint
- **Summary:** Uses rule-based systems as base classifiers in a dynamic feature selection framework, enhancing interpretability while formalizing additional sources of uncertainty in DFS. Predictions remain transparent through rule-based reasoning, unlike neural-based DFS methods. Relevant to DSGD++ because both leverage rule-based interpretability within an uncertainty-aware framework.
- **Link:** https://arxiv.org/abs/2508.02566

---

## 5. Alternatives to Dempster's Rule of Combination

### 5.1 Classical Alternative Rules

**[P45] Yager, R.R. "On the Dempster-Shafer Framework and New Combination Rules." Information Sciences, 41(2), 93-137 (1987).**
- **Venue:** Information Sciences
- **Summary:** Proposes Yager's rule, which assigns conflicting mass to the universal set (ignorance) rather than redistributing it via normalization as Dempster's rule does. Considered "epistemologically honest" as it does not modify evidence. The degree of ignorance reflects the mass of the universal set. A key alternative that avoids the counterintuitive results of Dempster's rule under high conflict.

**[P46] Dubois, D., Prade, H. "On the Unicity of Dempster Rule of Combination." International Journal of Intelligent Systems, 1(2), 133-142 (1986).**
- **Venue:** Int. J. Intelligent Systems
- **Summary:** Introduces the disjunctive rule of combination, which is both associative and more robust than Dempster's rule in the presence of conflicting evidence. Appropriate when conflict may be due to poor source reliability. Provides theoretical underpinnings for understanding when Dempster's rule is and isn't appropriate.

**[P47] Smarandache, F., Dezert, J. "Proportional Conflict Redistribution Rules for Information Fusion." In: Advances and Applications of DSmT for Information Fusion (2004-2015).**
- **Venue:** Multiple publications (arXiv, UNM Digital Repository)
- **Summary:** Proposes five versions of Proportional Conflict Redistribution (PCR1-PCR5) rules with increasing precision. PCR5 and PCR6 were developed for managing partial conflicts precisely. Recent improvements (PCR5+, PCR6+) compute binary-keeping indexes to redistribute only focal elements that play an effective role in partial conflict. Offers the most sophisticated conflict redistribution among known combination rules.
- **Link:** https://arxiv.org/abs/cs/0408010

### 5.2 Flexible and Weighted Combination Rules

**[P48] Ma, M. et al. "A Flexible Rule for Evidential Combination in Dempster-Shafer Theory of Evidence." Applied Soft Computing, 85, 105512 (2019).**
- **Venue:** Applied Soft Computing (Elsevier)
- **Summary:** Proposes a flexible combination rule based on complete conflict and evidence weights. Can reallocate conflicting belief according to similarity after applying Dempster's rule. Addresses issues of unsatisfying associative properties and high computational complexity found in modified combination rules.
- **Link:** https://www.sciencedirect.com/science/article/abs/pii/S1568494619302820

**[P49] Murphy, C.K. "Combining Belief Functions When Evidence Conflicts." Decision Support Systems, 29, 1-9 (2000).**
- **Venue:** Decision Support Systems
- **Summary:** Proposes computing the arithmetic average of n pieces of evidence before applying Dempster's rule. Simple but influential approach to handling conflicting evidence. Limitation: assigns equal weight to all evidence regardless of conflict levels. Has been the basis for many subsequent weighted averaging methods.

**[P50] Lefevre, E., Colot, O., Vannoorenberghe, P. "Belief Function Combination and Conflict Management." Information Fusion, 3(2), 149-162 (2002).**
- **Venue:** Information Fusion (Elsevier)
- **Summary:** Comprehensive comparison of combination operators in evidence theory. Introduces operators allowing arbitrary redistribution of conflicting mass on propositions. Proposes a taxonomy of combination rules and provides theoretical analysis of their properties. Foundational reference for understanding the space of possible combination rules.
- **Link:** https://www.semanticscholar.org/paper/Belief-function-combination-and-conflict-management-Lefevre-Colot/7598a1fcacb31042ecc8a1913ca09af19a01d681

### 5.3 Recent Advances in Combination Rules

**[P51] Bai, S., Li, L., Chen, X. "Conflicting Evidence Combination Based on Belief Mover's Distance." Journal of Intelligent & Fuzzy Systems, 42(3) (2022).**
- **Venue:** J. Intelligent & Fuzzy Systems
- **Summary:** Proposes a novel conflict management approach using Belief Mover's Distance, a metric inspired by the Earth Mover's Distance from optimal transport theory. Provides a principled way to measure and manage conflict between evidence sources before combination.

**[P52] Chen, Z. et al. "Weighted Evidence Combination Rule Based on Evidence Distance and Uncertainty Measure." Mathematical Problems in Engineering, 5858272 (2018).**
- **Venue:** Mathematical Problems in Engineering
- **Summary:** Proposes weighted combination where weights are derived from both evidence distance (measuring conflict) and an uncertainty measure. Assigns lower weights to evidence that is highly conflicting or highly uncertain before applying Dempster's rule. Relevant to DSGD++ because both works use confidence/uncertainty measures to weight evidence.

**[P53] Li, Z. et al. "An Improved Conflicting-Evidence Combination Method Based on the Redistribution of the Basic Probability Assignment." Applied Intelligence, 52, 5765-5783 (2022).**
- **Venue:** Applied Intelligence (Springer)
- **Summary:** Proposes redistribution of BPA before combination to handle conflicting evidence. The redistribution is based on measuring the degree of conflict between sources and adjusting mass values accordingly. Conceptually related to DSGD++'s confidence-based initialization, which also adjusts mass distribution based on data characteristics.
- **Link:** https://link.springer.com/article/10.1007/s10489-021-02404-4

---

## 6. Additional Relevant Works

### 6.1 Multimodal Fusion with DST

**[P54] Huang, L., Ruan, S., Decazes, P., Denoeux, T. "Deep Evidential Fusion with Uncertainty Quantification and Reliability Learning for Multimodal Medical Image Segmentation." Information Fusion, 113, 102648 (2025).**
- **Venue:** Information Fusion (Elsevier)
- **Summary:** Deep framework for multimodal medical image segmentation where features from each modality are mapped to DS mass functions, corrected by contextual discounting with learned reliability coefficients, and combined via Dempster's rule. Tested on PET-CT lymphoma and multi-MRI brain tumor segmentation. The learned reliability coefficients provide insight into each modality's contribution.
- **Link:** https://arxiv.org/abs/2309.05919

**[P55] Al-Anbagi, H. et al. "An Adaptive and Late Multifusion Framework in Contextual Representation Based on Evidential Deep Learning and Dempster-Shafer Theory." Knowledge and Information Systems, 66, 4673-4719 (2024).**
- **Venue:** Knowledge and Information Systems (Springer)
- **Summary:** Proposes an adaptive multimodal fusion framework using EDL guided by DST. Addresses modality-context-based fusion challenges for multi-object classification. Achieves 98.45% accuracy in multimodality experiments across multiple contexts.
- **Link:** https://link.springer.com/article/10.1007/s10115-024-02150-2

### 6.2 DST Surveys and Systematic Reviews

**[P56] Liu, H. et al. "Dempster-Shafer Theory-Based Information Fusion for Natural Disaster Emergency Management: A Systematic Literature Review." Information Fusion, 109, 102431 (2024).**
- **Venue:** Information Fusion (Elsevier)
- **Summary:** Systematic review of 62 papers on DST's role in natural disaster emergency management. Covers DST's use for rapid response, situation assessment, resource allocation, and decision-making under uncertainty. Demonstrates the breadth of DST applications beyond classification.
- **Link:** https://www.sciencedirect.com/science/article/abs/pii/S1566253524003634

**[P57] The Dempster-Shafer Theory to Handle Uncertainty in Artificial Intelligence. In: Springer (2023).**
- **Venue:** Springer book chapter
- **Summary:** Book chapter providing a modern overview of DST applications in AI, covering classification, decision support, and uncertainty management. Discusses advantages of DST including its ability to represent multiple types of uncertainty in a human-understandable way.
- **Link:** https://link.springer.com/content/pdf/10.1007/978-981-99-5333-2_3.pdf

---

## Summary of Key Gaps and Opportunities for DSGD++

Based on this literature review, several observations emerge regarding the positioning of DSGD++:

1. **MAF Initialization is Understudied.** While there is extensive work on combination rules and uncertainty measures, the specific problem of how to initialize mass functions before optimization has received limited attention. Penafiel (2020) uses random initialization, Sedlacek & Bartos (2024) use statistical distributions, and DSGD++ proposes clustering-based initialization. This is a genuine gap in the literature.

2. **The Harmonic Mean Uncertainty Measure is Novel.** The extensive work on uncertainty measures in DST ([P31]-[P38]) focuses on global measures (entropy, belief intervals). DSGD++'s approach of combining (1-uncertainty) with the mass ratio via harmonic mean is a rule-level diagnostic that has no direct precedent in the surveyed literature.

3. **Bridge Between Classical DST and Deep Learning.** DSGD++ occupies an interesting position: it uses gradient descent optimization (like evidential deep learning) but maintains the interpretable rule-based structure of classical DST. Most recent work either goes fully neural ([P9]-[P15]) or stays purely rule-based ([P1]). DSGD++ represents a middle ground.

4. **Potential Extensions Identified:**
   - Incorporating rule-mining algorithms (RIPPER, C5.0, SkopeRules) as the paper suggests
   - Exploring alternative combination rules ([P45]-[P53]) instead of Dempster's rule
   - Connecting with conformal prediction ([P39]) for guaranteed confidence
   - Extending to evidential random forest framework ([P18])
   - Applying the confidence-based initialization to other DST architectures ([P11], [P19])

---

## Sources

- [Penafiel et al. 2020 - DSGD Classifier](https://www.sciencedirect.com/science/article/abs/pii/S0957417420300877)
- [Penafiel Thesis 2021](https://repositorio.uchile.cl/bitstream/handle/2250/177533/Interpretable-method-for-general-classification-using-Dempster-Shafer-theory.pdf)
- [Penafiel et al. 2021 - Stroke Risk](https://ieeexplore.ieee.org/document/9306826/)
- [DSGD++ - Tarkhanyan & Harutyunyan 2025](https://lib.jucs.org/article/164745/)
- [Sensoy et al. 2018 - Evidential Deep Learning (NeurIPS)](https://arxiv.org/abs/1806.01768)
- [Denoeux 2019 - Logistic Regression and DST](https://www.sciencedirect.com/science/article/abs/pii/S0950705119301558)
- [Tong, Xu, Denoeux 2021 - Evidential CNN Classifier](https://www.sciencedirect.com/science/article/abs/pii/S0925231221004525)
- [Denoeux 2022 - Evidential NN Regression with GRFNs](https://arxiv.org/abs/2208.00647)
- [Pichon et al. 2024 - r-ERBFN](https://link.springer.com/chapter/10.1007/978-3-031-76235-2_26)
- [Gong et al. 2024 - Evidential Deep Network for Large Datasets](https://normandie-univ.hal.science/hal-04448387v1)
- [Su, Hu, Denoeux 2021 - Distributed Rough Evidential K-NN](https://ieeexplore.ieee.org/document/9103960/)
- [Gong et al. 2022 - Belief Entropy Tree and Random Forest](https://www.mdpi.com/1099-4300/24/5/605)
- [Hoarau et al. 2023 - Evidential Random Forests](https://www.sciencedirect.com/science/article/abs/pii/S0957417423011545)
- [Kurniawan & Mambo 2024 - DST Feature Fusion for CNNs](https://arxiv.org/abs/2405.20230)
- [Yaghoubi et al. 2022 - Multi-Classifier DST Fusion](https://journals.sagepub.com/doi/abs/10.1177/14759217211007130)
- [Zhao et al. 2022 - DST Survey](https://www.sciencedirect.com/science/article/abs/pii/S1568494622003696)
- [Chen et al. 2024 - EDL Survey](https://arxiv.org/abs/2409.04720)
- [Li et al. 2023 - Correlation Belief Function](https://www.nature.com/articles/s41598-023-34577-y)
- [Shiraishi et al. 2025 - DS-Fuzzy Classifier](https://dl.acm.org/doi/10.1145/3717613)
- [Mehravaran et al. 2025 - DDCAM](https://link.springer.com/content/pdf/10.1007/s11042-025-20866-3.pdf)
- [Jirousek & Shenoy 2023 - Uncertainty Measures Comparison](https://www.sciencedirect.com/science/article/pii/S0020025523006126)
- [Qin et al. 2020 - Improved Total Uncertainty](https://pmc.ncbi.nlm.nih.gov/articles/PMC7516972/)
- [Pan et al. 2020 - Belief Entropy Based on BPA and FOD](https://pmc.ncbi.nlm.nih.gov/articles/PMC7517227/)
- [Wang et al. 2023 - Belief Interval-Based Uncertainty](https://www.sciencedirect.com/science/article/abs/pii/S0020025523007351)
- [Huang et al. 2025 - Evaluation of Belief Entropies](https://link.springer.com/article/10.1007/s10462-025-11130-z)
- [Ulmer et al. 2024 - Conformal + DST](https://link.springer.com/chapter/10.1007/978-3-031-70344-7_6)
- [Shen et al. 2024 - EDL Mirage (NeurIPS)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c3177be226ee12e34d6ba3b5e6fe6a5b-Abstract-Conference.html)
- [Kohl et al. 2023 - DS-Inspired Active Learning Measures](https://link.springer.com/chapter/10.1007/978-3-031-50959-9_33)
- [Li et al. 2023 - BPA Generation with Cosine Similarity](https://www.nature.com/articles/s41598-023-35195-4)
- [McGuire et al. 2025 - DST Malware Detection](https://www.mdpi.com/2227-7390/13/16/2677)
- [Ma et al. 2019 - Flexible Combination Rule](https://www.sciencedirect.com/science/article/abs/pii/S1568494619302820)
- [Lefevre et al. 2002 - Combination Rules Survey](https://www.semanticscholar.org/paper/Belief-function-combination-and-conflict-management-Lefevre-Colot/7598a1fcacb31042ecc8a1913ca09af19a01d681)
- [Li et al. 2022 - Improved Conflicting Evidence Combination](https://link.springer.com/article/10.1007/s10489-021-02404-4)
- [Huang et al. 2025 - Deep Evidential Fusion for Segmentation](https://arxiv.org/abs/2309.05919)
- [Al-Anbagi et al. 2024 - Adaptive Multifusion EDL Framework](https://link.springer.com/article/10.1007/s10115-024-02150-2)
- [Liu et al. 2024 - DST Disaster Management Review](https://www.sciencedirect.com/science/article/abs/pii/S1566253524003634)
- [Valdivia et al. 2024 - Interpretable Clustering with DST](https://lib.jucs.org/article/164694/)
- [Hajiabadi et al. 2024 - Cloud Troubleshooting with DST](https://www.mdpi.com/2076-3417/14/3/1047)
- [Li 2022 - Interpretable Systems with Evidential Prospect Theory](https://link.springer.com/article/10.1007/s10489-022-03276-y)
- [Xu et al. 2025 - Dynamic Feature Selection Rule-Based](https://arxiv.org/abs/2508.02566)
