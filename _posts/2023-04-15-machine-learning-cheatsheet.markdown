---
layout: cheatsheet
title: Machine Learning Cheatsheet
description: Machine Learning Cheatsheet
image: /assets/images/machine-learning-cheatsheet/splash.png
authors: [tlyleung]
permalink: machine-learning-cheatsheet
---

<section markdown="1" style="background-image: url('/assets/images/streamline/brain-1.svg');">
# Machine Learning Cheatsheet

This cheatsheet attempts to give a high-level overview of the incredibly large field of Machine Learning. Please [contact me](/authors/tlyleung) for corrections/omissions.

*Last updated: 1 January 2024*
</section>

<section markdown="1" style="background-image: url('/assets/images/streamline/official-building-3.svg');">
# Background

- Artificial Intelligence is the ability of a machine to demonstrate human-like intelligence.
- Machine Learning is the field of study that gives computers the ability to learn without explicitly being programmed.
- Machine Learning has become possible because of:
  - Massive labelled datasets, e.g. ImageNet
  - Improved hardware and compute, e.g. GPUs
  - Algorithms advancements, e.g. backpropagation
</section>

<section markdown="1" style="background-image: url('/assets/images/streamline/synchronize-arrows-three.svg');">
# Machine Learning Lifecycle

1. Problem Framing
2. Data Assembly
3. Model Training
4. Model Evaluation
5. Model Deployment
</section>

<section markdown="1" style="background-image: url('/assets/images/streamline/question-help-circle.svg');">
# Problem Framing

- State the goal of the system
- Define success metrics
- Verify you have the data needed to train a model
- Identify the model's inputs and outputs
- Detail the constraints of the system[^cs329s]
  - model performance
  - usefulness threshold
  - false negatives vs. false positives
  - inference latency
  - inference cost
  - interpretability
  - freshness requirements and training frequency
  - online or batch
    - online: generate predictions after requests arrive, e.g. speech recognition
    - batch: generate predictions periodically before requests arrive, e.g. Netflix recommendations
  - cloud vs. edge vs. hybrid
    - cloud: no energy, power or memory constraints
    - edge: can work without unreliable connections, no network latency, fewer privacy concerns, cheaper
    - hybrid: common predictions are precomputed and stored on device
  - peak requests
  - number of users
  - confidence measurement (if confidence below threshold, discard, clarify or refer to humans)
  - privacy
    - annotation: can data be shipped to outside organisations?
    - storage: what data are you allowed to store and for how long?
    - third-party: can you share data with a third-party?
    - regulations
</section>

<section markdown="1" style="background-image: url('/assets/images/streamline/database-2.svg');">
# Data Assembly

## Workflow

1. Data Collection
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing
4. Feature Engineering
5. Feature selection: remove features with low variance, recursive feature elimination, sequential feature selection
6. Sampling Strategy: sequential, random, subset, weighted
7. Data Splits: train-test-validation split, windows splitting of time series data

---

## Data Collection[^data_checklist]

Good data should:
- have good predictive power (an expert should be able to make a correct prediction with the data)
- have very little missing values (when missing values do occur, they should be explainable and occur randomly)
  - Missing Not At Random (MNAR): missing due to the value itself
  - Missing At Random (MAR): missing due to another observed variable
  - Missing Completely At Random (MCAR): no pattern to which values are missing
- be labelled
- be correct and accurate
- be documented
- be unbiased

---

## Data Biases[^cs329s]

- Sampling/selection bias
- Under/over representation of subgroups
- Human biases embedded in the data
- Labelling bias
- Algorithmic bias

---

## Data Labelling[^cs329s]

- Hand-labelling, data lineage (track where data/labels come from)
- Use Labelling Functions (LFs) to label training data programmatic using different heuristics, including pattern matching, boolean search, database lookup, prediction from legacy system, third-party model, crowd labels. LFs can be noisy, correlated and conflicting
- Weak supervision, semi supervision, active learning, transfer learning

---

## Class Imbalance[^cs329s]

- Collect more data
- Data-level methods
  - Undersample majority class (can cause overfitting), e.g. Tomek Links makes decision boundaries clearer by finding pairs of close samples from opposite classes and removes the majority sample 
  - Oversample minority class (can cause loss of information), e.g. generate synthetic minority oversampling (SMOTE)
- Algorithm-level methods
  - Cost-sensitive learning penalises the misclassification of minority class samples more heavily than majority class samples
  - Class-balance loss by giving more weight to rare classes
  - Focal loss by giving more weight to difficult samples

---

## Data Preprocessing[^sklearn]

- **Missing Data**
  - Collect more data
  - Drop row/column
  - Constant imputation
  - Univariate imputation: replace missing values with the column mean/median/mode
  - Multivariate imputation: use all available features to estimate missing values
  - Nearest neighbours imputation: use an euclidean distance metric to find nearest neighbors
  - Add missing indicator column
- **Structured Data**
  - Categorical: ordinal encoding, one-hot encoding
  - Numeric: discretisation, min-max normalisation, z-score normalisation (when variables follow a normal distribution), log scaling (when variables follow an exponential distribution), power transform (mapping to Gaussian distribution using Yeo-Johnson or Box-Cox transforms)
- **Unstructured Data**
  - Audio:
  - Images: decode, resize, normalise
  - Text: normalisation (lower-casing, punctuation removal, strip whitespaces, strip accents, lemmatisation and stemming), tokenisation, token to IDs
  - Videos:
- **Dimensionality Reduction**
  - Principal Component Analysis (PCA): find a subset of features that capture the variance of the original features
  - Feature agglomeration: use Hierarchical Clustering to group features that behave similarly
- **Feature Crossing**
  - Combine two or more features to create a new feature
- **Positional Embeddings**
  - Can be either learned or fixed (Fourier features)

---

## Data Leakage[^cs329s]

- Splitting time-correlated data randomly instead of by time
- Preprocessing data before splitting, e.g. using the whole dataset to generate global statistics like the mean and using it to impute missing values
- Poorly handling of data duplication before splitting
- Group leakage, group of examples have strongly correlated labels but are divided into different splits
- Leakage from data collection process, e.g. doctors sending high-risk patients to a better scanner
- Detect data leakage by measuring correlation of a feature with labels, feature ablation study, monitoring model performance when new features are added
---

## Feature Engineering

- **Events:** event price, 
- **Location:** walk score, transit score, same region
- **Social:** how many people attending event, attendance by friends, invited by other user?, hosted by a friend?, attendance of events by same host
- **Time:** remaining time until event begins, estimated travel time
- **User:** age, gender
</section>

<section markdown="1" style="background-image: url('/assets/images/streamline/server-3.svg');">
# Model Training: Overview

## Workflow

1. Decide whether to train from scratch or fine-tune existing model
2. Choose loss function
3. Establish a simple baseline
4. Experiment with simple models
5. Switch to more complex models
6. Use an ensemble of models

---

## Key Concepts

- **Curse of Dimensionality:** As the number of features in a dataset increases, the volume of the feature space increases so fast that the available data becomes sparse. This makes it hard to have enough data to give meaningful results, leading to overfitting.
- **Overfitting and Underfitting:** overfitting occurs when a model learns the training data too well and can't generalise to unseen data, while underfitting happens when a model isn't powerful enough to model the training data.

  <div class="row">
    <div class="col-4 text-center"></div>
    <div class="col-4 text-center">Classification</div>
    <div class="col-4 text-center">Regression</div>
  </div>
  <div class="row">
    <div class="col-4 d-flex align-items-center">
      Underfit
    </div>
    <div class="col-4">
      <img src="/assets/images/machine-learning-cheatsheet/classification-underfit.svg" alt="Classification Underfit" class="img-fluid">
    </div>
    <div class="col-4">
      <img src="/assets/images/machine-learning-cheatsheet/regression-underfit.svg" alt="Regression Underfit" class="img-fluid">
    </div>
  </div>
  <div class="row">
    <div class="col-4 d-flex align-items-center">
      Good Fit
    </div>
    <div class="col-4">
      <img src="/assets/images/machine-learning-cheatsheet/classification-good-fit.svg" alt="Classification Good Fit" class="img-fluid">
    </div>
    <div class="col-4">
      <img src="/assets/images/machine-learning-cheatsheet/regression-good-fit.svg" alt="Regression Good Fit" class="img-fluid">
    </div>
  </div>
  <div class="row">
    <div class="col-4 d-flex align-items-center">
      Overfit
    </div>
    <div class="col-4">
      <img src="/assets/images/machine-learning-cheatsheet/classification-overfit.svg" alt="Classification Overfit" class="img-fluid">
    </div>
    <div class="col-4">
      <img src="/assets/images/machine-learning-cheatsheet/regression-overfit.svg" alt="Regression Overfit" class="img-fluid">
    </div>
  </div>

- **Bias and Variance:** Bias refers to the error introduced by approximating a real-world problem with a simplified model. High bias can cause an algorithm to miss relevant relations between features and target outputs (underfitting). Variance refers to the amount by which a model would change if estimated using a different training dataset. High variance can cause an algorithm to model random noise in the training data, not the intended outputs (overfitting).

  <div class="row">
    <div class="col-4 text-center"></div>
    <div class="col-4 text-center">Low Bias</div>
    <div class="col-4 text-center">High Bias</div>
  </div>
  <div class="row">
    <div class="col-4 d-flex align-items-center">Low Variance</div>
    <div class="col-4">
      <img src="/assets/images/machine-learning-cheatsheet/bias-low-variance-low.svg" alt="Low Bias, Low Variance" class="img-fluid">
    </div>
    <div class="col-4">
      <img src="/assets/images/machine-learning-cheatsheet/bias-high-variance-low.svg" alt="High Bias, Low Variance" class="img-fluid">
    </div>
  </div>
  <div class="row">
    <div class="col-4 d-flex align-items-center">High Variance</div>
    <div class="col-4">
      <img src="/assets/images/machine-learning-cheatsheet/bias-low-variance-high.svg" alt="Low Bias, High Variance" class="img-fluid">
    </div>
    <div class="col-4">
      <img src="/assets/images/machine-learning-cheatsheet/bias-high-variance-high.svg" alt="High Bias, High Variance" class="img-fluid">
    </div>
  </div>

- **Bias-Variance Trade-off:** As you increase the complexity of your model, you will typically decrease bias but increase variance. On the other hand, if you decrease the complexity of your model, you increase bias and decrease variance. 
- **Vanishing/Exploding Gradients:** When training a deep neural network, if the gradient values become very small, they get "squashed" due to the activation functions resulting in vanishing gradients. When these small values get multiplied during backpropagation they can become near zero, which results in a lack of updates to the network weights and the training stalling. On the other hand, if the gradients become too large, they "explode", causing model weights to update too drastically and making model training unstable.
- **Universal Approximation Theorem:** A neural network with a single hidden layer can approximate any continuous function for inputs within a specific range
- **Learning Curve:** Model performance as a function of number of training examples, can be good for estimating if performance can be improved with more data

---

## Debugging

- Overfit model on a subset of data
- Look out for exploding gradients (use gradient clipping)
- Turn on `detect_anomaly` so that any backward computation that generates `NaN` will raise an error.

---

## Performance Tuning[^pytorch][^pytorch_lightning]
- Enable asynchronous data loading and augmentation using `num_workers > 0` and `pin_memory = True`
- Disable bias for convolutions before batch norms
- Use learning rate scheduler
- Use mixed precision
- Accumulate gradients by running a few small batches before doing a backward pass
- Saturate GPU by maxing-out batch size (downside: higher batch sizes may cause training to get stuck in local minima)
- Use Distributed Data Parallel (DDP) for multi-GPU training
- Clip gradients to avoid exploding gradients
- Disable gradient calculation for val/test/predict

---

## Hyperparameter Optimisation (HPO)

- **Grid search:** exhaustively search within bounds
- **Random search:** randomly search within bounds
- **Bayesian search:** modeled as Gaussian process

---

## Neural Architecture Search[^cs329s] (NAS)

- **Search Space:** set of operations, (e.g. convolutions, fully-connected layers, pooling, etc.) and how they can be connected
- **Search Strategy:** random, reinforcement learning, evolution

---

## Cross-validation[^sklearn] (CV)

- **K-fold:** divide samples into $$k$$ folds; train model on $$k-1$$ folds and evaluate using the left out fold 
- **Leave One Out (LOO):** train model on all samples except one and evaluate using the left out sample
- **Stratified K-fold:** similar to K-fold, but each fold contains the same class balance as the full dataset
- **Group K-fold:** similar to K-fold, but ensure that groups (samples from the same data source) do not span different folds
- **Time Series Split:** to ensure only past observations are used to predict future observations, train model on first $$n$$ folds and evaluate on the $$n+1$$-th fold
</section>

<section markdown="1" style="background-image: url('/assets/images/streamline/trends-hot-flame.svg');">
# Model Training: PyTorch[^pytorch]

## Optimisers
- Adam
- Momentum
- RMSProp
- Stochastic Gradient Descent (SGD)

---

## Initialisations
- Kaiming
- Xavier

---

## Regularisation
- Augmentation
  - Image: random crop, saturation, flip, rotation, translation, perturb using random noise
  - Text: swap with synonyms, add degree adverbs, perturb with random word replacements
- Data synthesis
  - Image: mixup (inputs and labels are linear combination of multiple classes)
  - Text: template-based, language model-based
- Dropout
- Early stopping
- L1 regularisation (Lasso) tends to lead to sparsity because it penalises the absolute value of the weights, thereby encouraging some 
weights to go to zero (it can also be used for feature selection).
- L2 regularisation (Ridge) penalises the square of the weights, thereby pushing them closer to zero but not necessarily to zero. This leads to models that are less sparse and can better manage multicollinearity.

---

## Pooling
- Average pool
- Min pool
- Max pool

---

## Normalisation

<div class="row">
  <div class="col-3 d-flex align-items-center">
    Batch Norm
  </div>
  <div class="col-3">
    <img src="/assets/images/machine-learning-cheatsheet/norm-batch.svg" alt="Batch Norm" class="img-fluid">
  </div>
  <div class="col-3 d-flex align-items-center">
    Layer Norm
  </div>
  <div class="col-3">
    <img src="/assets/images/machine-learning-cheatsheet/norm-layer.svg" alt="Layer Norm" class="img-fluid">
  </div>
</div>
<div class="row">
  <div class="col-3 d-flex align-items-center">
    Group Norm
  </div>
  <div class="col-3">
    <img src="/assets/images/machine-learning-cheatsheet/norm-group.svg" alt="Group Norm" class="img-fluid">
  </div>
  <div class="col-3 d-flex align-items-center">
    Instance Norm
  </div>
  <div class="col-3">
    <img src="/assets/images/machine-learning-cheatsheet/norm-instance.svg" alt="Instance Norm" class="img-fluid">
  </div>
</div>

---

## Activations

<div class="row">
  <div class="col-3 d-flex align-items-center">
    Sigmoid<br />1 / (1 + e<sup>-x</sup>)
  </div>
  <div class="col-3">
    <img src="/assets/images/machine-learning-cheatsheet/activation-sigmoid.svg" alt="Sigmoid" class="img-fluid">
  </div>
  <div class="col-3 d-flex align-items-center">
    ReLU<br />max(0,x)
  </div>
  <div class="col-3">
    <img src="/assets/images/machine-learning-cheatsheet/activation-relu.svg" alt="ReLU" class="img-fluid">
  </div>
</div>
<div class="row">
  <div class="col-3 d-flex align-items-center">
    Tanh<br />tanh(x)
  </div>
  <div class="col-3">
    <img src="/assets/images/machine-learning-cheatsheet/activation-tanh.svg" alt="Tanh" class="img-fluid">
  </div>
  <div class="col-3 d-flex align-items-center">
    Leaky ReLU<br />max(0.1x,x)
  </div>
  <div class="col-3">
    <img src="/assets/images/machine-learning-cheatsheet/activation-leaky-relu.svg" alt="Leaky ReLU" class="img-fluid">
  </div>
</div>

---

## Loss Functions

- **Cross-Entropy** measures how close the predicted probability distribution is with the true distribution

  $$l_n = - w_{y_n} \log \frac{\exp(x_{n,y_n})}{\sum_{c=1}^C \exp(x_{n,c})}$$
 
- **Connectionist Temporal Classification (CTC)** is used where the alignment between input and output sequences is unknown

- **Kullback–Leibler (KL) Divergence** measures of how one probability distribution diverges or is different from a second, expected probability distribution.

  $$L(y_{\text{pred}},\ y_{\text{true}})  = y_{\text{true}} \cdot (\log y_{\text{true}} - \log y_{\text{pred}})$$

- **Mean Absolute Error (L1)** measures
 
  $$l_n = \left| x_n - y_n \right|$$

- **Mean Squared Error (Squared L2 Norm)** measures the average squared difference between the estimated values and the actual value

  $$l_n = \left( x_n - y_n \right)^2$$

- **Negative Log Likelihood (NLL)** measures the disagreement between the true labels and the predicted probability distributions,

  $$l_n = - w_{y_n} x_{n,y_n}$$

---

## Distributed Training

- Data parallelism: split the data across devices so that each device sees a fraction of the batch
- Model parallelism: split the model across devices so that each device runs a fragment of the model


</section>

<section markdown="1" style="background-image: url('/assets/images/streamline/earth-2.svg');">
# Model Evaluation: Responsible AI

## Fairness

- Slice-based evaluation, e.g. when working with website traffic, slice data among: gender, mobile vs. desktop, browser, location
- Check for consistency over time
- Determine slices by heuristics or error analysis

---

## Explainability

- **Integrated Gradients:** compute the contribution of each feature to a prediction by integrating gradients over the path from the baseline
- **LIME (Local Interpretable Model-agnostic Explanations):** creates a simpler, interpretable model around a single prediction to explain how the model behaves at that specific instance.
- **Sampled Shapley:** estimates the contribution of each feature by averaging over subsets of features sampled from the input data.
- **SHAP (SHapley Additive exPlanations):** assigns each feature an importance value for a particular prediction, based on the concept of Shapley values from cooperative game theory
- **XRAI (eXplanation with Ranked Area Integrals):** segments an input image and ranks the segments based on their contribution to the model's prediction

---

## Compactness

- Reduces memory footprint and increases computation speed
- Quantisation: reduce model size by using fewer bits to represent parameters 
- Knowledge distillation: train a small model (student) to mimic the results of a larger model (teacher)
- Pruning: remove nodes or set least useful parameters to zero
- Low-ranked factorisation: replace convolution filters with compact blocks

---

## Robustness


- **Determinism Test:** ensure same outputs when predicting using same model
- **Retraining Invariance Test:** ensure similar outputs when predicting using re-trained model
- **Perturbation Test:** ensure small changes to numeric inputs don't cause big changes to outputs
- **Input Invariance Test:** ensure changes to certain inputs don't cause changes to outputs
- **Directional Expectation Test:** ensure changes to certain inputs cause predictable changes to outputs
- **Ablation Test:** ensure all parts of the model are relevant for model performance
- **Fairness Test:** ensure different slices have similar model performance
- **Model Calibration Test:** ensure events should happen according to the proportion predicted

---

## Safety

- **Alignment:**
- **Red Teaming:** experts simulate potential attacks on a system to identify vulnerabilities, test defenses, and improve system security before actual attackers do
</section>

<section markdown="1" style="background-image: url('/assets/images/streamline/analytics-pie-2.svg');">
# Model Evaluation: Metrics


## Offline Metrics (Before Deployment)

- Baseline
  - Predict at random (uniformly or following label distribution)
  - Zero rule baseline (always predict the most common class)
  - Simple heuristics
  - Human baseline
  - Existing solutions
- Classification
  - Confusion Matrix

    |         | Class 1             | Class 2             |
    |---------|---------------------|---------------------|
    | Class 1 | True-positive (TP)  | False-positive (FP) |
    | Class 2 | False-negative (FN) | True-negative (TN)  |
    {: .table .table-striped }

  - Type I error: FP
  - Type II error: FN
  - Precision: TP / (TP + FP), i.e. a classifier's ability not to label a negative sample as positive
  - Recall or True-positive rate: TP / (TP + FN), i.e. a classifier's ability to find all positive samples
  - False-positive rate: FP / (FP + TN), i.e. a classifier's inability to find all negative samples
  - F1 score (harmonic mean of precision and recall): 2 × precision × recall / (precision + recall)
  - Precision-recall curve: trade-off between precision and recall, a higher PR-AUC indicates a more accurate model
  - Receiver operator characteristic (ROC) curve: trade-off between true-positive rate (recall) and false-positive rate, a higher ROC-AUC indicates a model better at distinguishing positive and negative classes
- Regression
  - Mean squared error (MSE): average of the squared differences between the predicted and actual values, emphasising larger errors
  - Mean absolute error (MAE): average of the absolute differences between the predicted and actual values, treating all errors equally
  - Root mean square error (RMSE): square root of the MSE, providing error in the same units as the predicted and actual values and emphasizing larger errors like MSE
- Object Recognition
  - Intersection over union (IOU): ratio of overlap area with union area
- Ranking
  - Recall@k: proportion of relevant items that are included in the top-k recommendations
  - Precision@k: proportion of top-k recommendations that are relevant
  - Mean reciprocal rank (MRR): $$\frac{1}{m} \sum_{i=1}^m \frac{1}{\textrm{rank}_i}$$, i.e. where is the first relevant item in the list of recommendations?
  - Hit rate: how often does the list of recommendations include something that's actually relevant?
  - Mean average precision (mAP): mean of the average precision scores for each query
  - Diversity: measure of how different the recommended items are from each other
  - Coverage: what's the percentage of items seen in training data that are also seen in recommendations?
  - Cumulative gain (CG): $$\sum_{i=1}^p rel_i$$, i.e. sum of relevance scores obtained by a set of recommendations
  - Discounted cumulative gain (DCG): $$\sum_{i=1}^p \frac{\textrm{rel}_i}{\log_2(i+1)}$$, i.e. CG discounted by position
  - Normalised discounted cumulative gain (nDCG): $$\frac{\textrm{DCG}_p}{\textrm{IDCG}_p}$$, i.e. extension of CG that accounts for the position of the recommendations (discounting the value of items appearing lower in the list), normalised by maximum possible score
- Image Generation
  - FID
  - Inception score
- Natural Language
  - BLEU
  - Perplexity: average "branching factor" per token

---

## Online Metrics (After Deployment)

- Model's impact of user behavior or system performance
- Event recommendation: conversion rate, bookmark rate, revenue lift
- Safety: prevalence, harmful impressions, valid appeals, proactive rate, user reports per harmful class
- Video recommendations: click-through-rate, video completion rate, total watch time, explicit user feedback
</section>

<section markdown="1" style="background-image: url('/assets/images/streamline/space-rocket-flying.svg');">
# Model Deployment

## Deployment Strategies (B to replace A)
- Recreate strategy: stop A, start B
- Ramped strategy: shift traffic from A to B behind same endpoint
- Blue/green: shift traffic from A to B using different endpoint

---

## Testing Strategies
- Canary: targeting small set of users with latest version
- Shadow: mirror incoming requests and route to shadow application
- A/B: route to new application depending on rules or contextual data
- Interleave: mix recommendations from A and B and see which recommendations are clicked on

---

## Model Monitoring
- Model monitoring is essential because while traditional software systems fail explicitly (error messages), Machine Learning systems fail silently (bad outputs)
- Operation-related metrics:
  - Latency
  - Throughput
  - Requests per minute/hour/day
  - Percentage of successful requests
  - CPU/GPU utilisation
  - Memory utilisation
  - Availability
- Machine Learning metrics[^cs329s]:
  - Feature and label statistics, e.g. mean, median, variance, quantiles, skewness, kurtosis, etc.
  - Task-specific online metrics


Continual Learning
- Continually adapt models to changing data distributions

---

## ML-specific Failures[^cs329s]

Train-serving skew is when a model performs well during development but poorly after production

- Upstream Drift: caused by a discrepancy between how data is handled in the training and serving pipelines (should log features at serving time)[^ml_rules]
- Data Distribution Shifts: model may perform well when first deployed, but poorly over time (can be sudden, cyclic or gradual)
  - Feature/covariate shift
    - Change in the distribution of input data, P(X), but relationship between input and output, P(Y\|X), remains the same
    - E.g. when predicting sales based on weather, if weather patterns change (e.g. more rainy days), but the relationship between weather and sales remains constant (e.g. rainy days always lead to fewer sales)
    - In training, can be caused by changes to data collection, while in production, can be caused by changes to environment
  - Label shift
    - Change in the distribution of output labels, P(Y), but relationship between output and input, P(X\|Y), remains the same
    - E.g. when predicting diseases, if a disease becomes more common, but symptoms for each disease remains constant
  - Concept drift
    - Change in the relationship between input and output, P(Y\|X), but the distribution of input data, P(X), remains the same
    - E.g. when predicting rain from cloud patterns, if the cloud patterns remain the same but their association with rain changes (maybe due to climate change)
- Degenerate Feedback Loops: when predictions influence the feedback, which is then used to extract labels to train the next iteration of the model
  - Recommender system example: originally, A is ranked marginally higher than B, so the model recommends A. After a while, A is ranked much higher than B. Can be detected using Average Recommended Popularity (ARP) and Average Percentage of Long Tail Items (APLT).
  - Resume screening example: originally, model thinks X is a good feature, so the model recommends resume with X. After a while, hiring managers only hires people with X and model confirms X is good. Can be mitigated using randomisation and positional features.
</section>

<section markdown="1" style="background-image: url('/assets/images/streamline/neural-swarm-2.svg');">
# Models: Supervised Learning

Supervised learning models make predictions after seeing lots of data with the correct answers. The model discovers the relationship between the data and the correct answers.

</section>

<section markdown="1" style="background-image: url('/assets/images/streamline/hierarchy-9.svg');">
# Models: Unsupervised Learning

Unsupervised learning involves finding patterns and structure in input data without any corresponding output data.

</section>

<section markdown="1" style="background-image: url('/assets/images/streamline/chess-figures.svg');">
# Models: Reinforcement Learning[^spinning_up]

</section>

<section markdown="1" style="background-image: url('/assets/images/streamline/common-file-text.svg');">
# Models: Language Models

</section>

<section markdown="1" style="background-image: url('/assets/images/streamline/rating-five-star.svg');">
# Models: Recommender Systems[^mlsystemdesign]

- Behavioural principles
  - Similar items are symmetric, e.g. white polo shirts
  - Complementary items are asymmetric, e.g. buying a television, suggest a HDMI cable

## Rule-based

---

## Embedding-based

- **Content-based Filtering**
  - Item feature similarities
  - Pros: ability to recommend new videos, ability the capture unique user interests
  - Cons: difficult to discover a user's new interests, requires domain knowledge to engineer features
- **Collaborative Filtering**
  - User-to-user similarities or item-to-item similarities
  - Pros: no domain knowledge needed, easy to discover users' new areas of interest, efficient
  - Cons: cold-start problem, cannot handle niche interests
- **Hybrid Filtering**
  - Parallel or sequential combination of content-based and collaborative filtering

---

## Learning-to-Rank

- **Point-wise:** model takes each item individually and learns to predict an absolute relevancy score
- **Pair-wise:** model takes two ranked items and learns to predict which item is more relevant (RankNet, LambdaRank, LambdaMART)
- **List-wise:** model takes optimal ordering of items and learns the ordering (SoftRank, ListNet, AdaRank)
</section>

<section markdown="1" style="background-image: url('/assets/images/streamline/hierarchy-4.svg');">
# Models: Ensembles

- **Bagging (Bootstrap Aggregation):** reduces model variance by training identical models in parallel on different data subsets (random forests)
  - Pros: reduces overfitting, parallel training means little increase in training/inference time
  - Cons: not helpful for underfit models
- **Boosting:** reduces model bias and variance by training several weak classifiers sequentially (Adaboost, XGBoost)
  - Pros: reduces bias and variance
  - Cons: slower training and inference
- **Stacking (Stacked Generalisation):** reduces model bias and variance by training different models in parallel on the same dataset and using a meta-learner model to combine the results
  - Pros: reduces bias and variance, parallel training means little increase in training/inference time
  - Cons: prone to overfitting
- **Multiple Objective Optimisation (MOO)** combines results of different models with weightings; decouple models with different objectives for easier training, tweaking and maintenance

</section>

[^data_checklist]: [Is My Data Any Good? A Pre-ML Checklist.](https://services.google.com/fh/files/blogs/data-prep-checklist-ml-bd-wp-v2.pdf)
[^ml_rules]: [Rules of Machine Learning: Best Practices for ML Engineering.](https://developers.google.com/machine-learning/guides/rules-of-ml)
[^sklearn]: [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
[^cs329s]: [CS329S: Machine Learning Systems Design](https://stanford-cs329s.github.io/)

[^cs324]: [CS324: Large Language Models](https://stanford-cs324.github.io/winter2022/)
[^pair]: [People + AI Guidebook](https://pair.withgoogle.com/guidebook/)
[^mlinterviews]: [Machine Learning Interviews Book](https://huyenchip.com/ml-interviews-book/)
[^pytorch]: [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
[^pytorch_lightning]: [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)

[^coursera]: [Coursera Deep Learning Specialisation](https://www.coursera.org/specializations/deep-learning)
[^pandas]: [Pandas Documentation](https://pandas.pydata.org/docs/)
[^mlsystemdesign]: [Machine Learning System Design Interview](https://bytebytego.com/intro/machine-learning-system-design-interview)
[^google]: [Google Machine Learning Education](https://developers.google.com/machine-learning)
[^generativeai]: [Google Generative AI Learning Path](https://www.cloudskillsboost.google/paths/118)
[^hugging_face]: [Hugging Face Documentation](https://huggingface.co/docs)
[^cs229]: [CS3229: Machine Learning](https://cs229.stanford.edu/)
[^cs224n]: [CS3224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/)
[^cs231n]: [CS3314N: Deep Learning for Computer Vision](http://vision.stanford.edu/teaching/cs231n/)
[^reinforcement_learning]: [Introduction to Reinforcement Learning](https://www.deepmind.com/learning-resources/introduction-to-reinforcement-learning-with-david-silver)
[^spinning_up]: [Spinning Up in Deep Reinforcement Learning](https://spinningup.openai.com/en/latest/)
