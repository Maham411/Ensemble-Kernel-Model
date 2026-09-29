# Ensemble & Kernel Methods — Telco Customer Churn

## Overview

This project applies **Ensemble Learning and Kernel Methods** to the IBM Telco Customer Churn dataset.

The main goal is to predict whether a customer is likely to **churn (leave the company)** and compare different machine learning approaches:

* Gradient Boosting / XGBoost
* Linear SVM
* RBF SVM
* Hyperparameter tuning
* Early stopping
* Probability calibration
* Training and inference efficiency

The project focuses not only on classification performance, but also on whether predicted probabilities are useful for a real-world **customer retention and offer-budgeting system**.

---

# Dataset

The project uses the IBM Telco Customer Churn dataset.

* Approximately 7,000 customers
* Target variable: `Churn`
* `No` → 0
* `Yes` → 1

The `customerID` column was removed because it is an identifier and does not provide useful predictive information.

---

# Project Workflow

The complete workflow is:

```text
Raw Telco Customer Churn Dataset
            ↓
Remove customerID
            ↓
Convert Churn Yes/No → 1/0
            ↓
Train/Test Split
            ↓
Training Data
    ┌───────┴────────┐
    ↓                ↓
Training Set      Validation Set
    ↓                ↓
Model Training   Early Stopping
    │
    └───────────────┘
            ↓
Hyperparameter Tuning
            ↓
Final Tuned Models
            ↓
Untouched Test Set
            ↓
F1 / ROC-AUC / PR-AUC
            ↓
Calibration Analysis
            ↓
Brier Score
            ↓
Training & Inference Time
            ↓
Final Model Comparison
```

## Important Train/Test Principle

The project creates its **own train/test split from the raw dataset**.

The test set is kept separate from model training and hyperparameter tuning.

For early stopping, the training data was further divided into:

```text
Original Training Data
        ↓
Training Subset + Validation Subset
```

The **validation set** was used for early stopping.

The **test set was not used for early stopping or model training**.

This prevents information from the final test set from leaking into the training process.

---

# Data Preprocessing

The preprocessing pipeline includes:

1. Removing `customerID`
2. Separating numerical and categorical features
3. Standardizing numerical features using `StandardScaler`
4. One-hot encoding categorical features
5. Using `handle_unknown="ignore"` in `OneHotEncoder`

The preprocessing is performed through a scikit-learn `ColumnTransformer` and model pipelines.

Using:

```python
OneHotEncoder(handle_unknown="ignore")
```

helps prevent errors when a category appears during transformation that was not present in the training data.

---

# Part A — Conceptual Questions

## A1. Calibration vs Discrimination

### Question

Why is calibration important, rather than only AUC, when predicted probabilities are used to automatically decide how much retention-offer budget to allocate?

### Answer

**AUC measures discrimination**, meaning how well the model ranks customers from lower to higher churn risk.

**Calibration measures whether the predicted probabilities match the real observed rates.**

For example:

```text
Predicted probability = 0.80
```

Ideally, customers receiving predictions around 0.80 should actually churn at a rate close to 80%.

For an automated retention-budget system, this matters because the business is using the probability itself to make financial decisions.

Therefore:

* AUC tells us whether the model ranks customers correctly.
* Calibration tells us whether the probability values can be trusted.

A model can have a good AUC but still produce poorly calibrated probabilities.

---

## A2. Rare Churn / Class Imbalance

### Question

If churn prevalence is only around 3%, why is setting `scale_pos_weight` not enough by itself?

### Answer

`scale_pos_weight` can help the model pay more attention to the minority class, but it does not solve every problem caused by very low churn prevalence.

Other issues can include:

* Distribution shift
* Data leakage
* Poor probability calibration
* Low precision
* Low recall
* Poor performance on the minority class

Metrics such as **PR-AUC** and **recall at a fixed precision** can provide more useful information when the positive class is very rare.

The model's predicted probabilities should also be checked for calibration.

---

## A3. Learning Rate / Shrinkage

### Question

What does the learning rate do in boosting?

### Answer

The learning rate controls how much each new tree contributes to the final model.

For example:

```text
Learning rate = 0.1
```

means each tree makes a relatively small correction to the current model.

A smaller learning rate usually requires more trees.

```text
Small learning rate
        ↓
Smaller corrections
        ↓
Usually more trees
        ↓
Slower training
        ↓
Can provide better control over learning
```

A simple way to understand it is:

> Learning rate controls the size of each step while the model is learning.

---

## A4. Gradient Boosting vs Random Forest

### Question

How is boosting different from Random Forest?

### Answer

Random Forest uses **bagging**.

The trees are mostly trained independently using different samples/features, and their predictions are combined.

Boosting works differently.

Boosting builds trees **sequentially**.

Each new tree tries to correct mistakes made by the previous model.

```text
Initial prediction
       ↓
Tree 1
       ↓
Find errors
       ↓
Tree 2 corrects errors
       ↓
Find remaining errors
       ↓
Tree 3 corrects them
       ↓
Final model
```

So:

* Random Forest → many relatively independent trees
* Boosting → trees are built sequentially to improve previous predictions

---

## A5. Kernel Trick

### Question

What is the kernel trick in SVM?

### Answer

The kernel trick allows SVM to work with data that may not be linearly separable without explicitly creating all the higher-dimensional features.

Instead of manually transforming the data into a higher-dimensional space, the kernel calculates similarities between data points as if that transformation had happened.

The RBF kernel is useful when relationships between features and the target are nonlinear.

In simple words:

> The kernel trick allows SVM to find nonlinear patterns without explicitly creating the high-dimensional feature space.

---

# Part B — Experiments

# B1. XGBoost Hyperparameter Tuning

RandomizedSearchCV was used to tune:

* `learning_rate`
* `n_estimators`
* `max_depth`

### Search Space

```python
learning_rate = [0.01, 0.05, 0.1, 0.2]
n_estimators = [100, 300, 500]
max_depth = [3, 4, 6, 8]
```

### Best Hyperparameters

```text
learning_rate = 0.1
n_estimators = 100
max_depth = 4
```

### Best Cross-Validation F1

```text
0.5862
```

### Test Results

```text
F1 Score  = 0.5774
ROC-AUC   = 0.8422
```

The cross-validation F1 and test F1 are reasonably close, which indicates that the tuned model's performance on unseen test data was broadly consistent with its validation performance.

---

# B2. Linear SVM vs RBF SVM

Two SVM models were tested:

### Linear SVM

```text
F1 = 0.5832
```

### RBF SVM

```text
F1     = 0.5571
ROC-AUC = 0.8082
```

The linear SVM achieved a slightly higher F1 score than the RBF SVM in this experiment.

The RBF SVM was also evaluated using probability predictions for calibration analysis.

---

# B3. Early Stopping

An XGBoost model was trained with:

```text
n_estimators = 1000
learning_rate = 0.05
max_depth = 4
early_stopping_rounds = 20
```

A validation set was created from the training data.

The test set was **not used for early stopping**.

### Result

```text
Best iteration = 110
```

This means the validation log-loss stopped improving sufficiently after around the 110th boosting iteration.

Instead of unnecessarily building all 1,000 trees, early stopping allowed the model to stop after the useful learning had been reached.

### Why is this useful?

It can:

* Reduce unnecessary computation
* Reduce training time
* Help prevent overfitting
* Automatically determine when additional trees are no longer helping the validation performance

---

# B4. Training and Inference Time

The models were compared on the same dataset and environment.

## Training Time

```text
XGBoost  = 0.593 seconds
RBF SVM  = 15.692 seconds
```

## Probability Inference Time

```text
XGBoost  = 0.0438 seconds
RBF SVM  = 0.5607 seconds
```

## Approximate Time Per Customer

```text
XGBoost  = 0.031 ms/customer
RBF SVM  = 0.398 ms/customer
```

These timings are specific to the experiment environment and dataset size.

They should not be interpreted as universal performance numbers for every hardware or production environment.

---

# Part C — Final Analysis

# C1. Final Tuned Hyperparameters

The final tuned XGBoost model used:

```text
n_estimators = 100
max_depth = 4
learning_rate = 0.1
```

These were selected through the RandomizedSearchCV process.

---

# C2. Probability Calibration

Calibration was evaluated because the final business use case involves using predicted probabilities for retention-offer budgeting.

## XGBoost Calibration Results

The mean predicted probabilities and observed churn rates were:

| Mean Predicted Probability | Actual Churn Rate |
| -------------------------: | ----------------: |
|                     0.0075 |            0.0071 |
|                     0.0192 |            0.0213 |
|                     0.0406 |            0.0567 |
|                     0.0812 |            0.0780 |
|                     0.1387 |            0.1773 |
|                     0.2196 |            0.2286 |
|                     0.3274 |            0.3688 |
|                     0.4498 |            0.3901 |
|                     0.5812 |            0.5532 |
|                     0.7822 |            0.7730 |

The predicted probabilities generally follow the observed churn rates, although some differences appear in the middle probability ranges.

For example:

```text
Predicted ≈ 0.782
Actual    ≈ 0.773
```

This is relatively close.

Another example:

```text
Predicted ≈ 0.450
Actual    ≈ 0.390
```

Here the model overestimated the observed churn rate.

Therefore, the model is **reasonably calibrated but not perfectly calibrated**.

---

## Brier Score

Brier Score measures the accuracy of probabilistic predictions.

**Lower is better.**

| Model   | Brier Score |
| ------- | ----------: |
| XGBoost |  **0.1365** |
| RBF SVM |      0.1460 |

XGBoost achieved the lower Brier score in this experiment.

This provides additional evidence that its predicted probabilities were more accurate overall than the RBF SVM probabilities.

### Important Production Consideration

The calibration results should not be interpreted as proof that the probabilities are perfectly calibrated.

If the probabilities are going to directly control significant financial decisions, an additional calibration step such as:

* Platt scaling / sigmoid calibration
* Isotonic regression

could be evaluated using a separate calibration/validation procedure.

---

# C3. Final Model Comparison

| Metric          |        XGBoost |    RBF SVM |
| --------------- | -------------: | ---------: |
| F1              |     **0.5774** |     0.5571 |
| ROC-AUC         |     **0.8422** |     0.8082 |
| Brier Score     |     **0.1365** |     0.1460 |
| Training Time   |  **0.593 sec** | 15.692 sec |
| Inference Time  | **0.0438 sec** | 0.5607 sec |
| Time / Customer |   **0.031 ms** |   0.398 ms |

---

# Final Model Selection

Based on the experimental results, **XGBoost was selected as the final model for this project**.

The reason is not based on a single metric.

XGBoost showed:

* Higher F1 score
* Higher ROC-AUC
* Lower Brier score
* Much lower training time
* Much lower probability inference time
* Generally reasonable calibration

### Performance

```text
XGBoost F1     = 0.5774
RBF SVM F1     = 0.5571

XGBoost ROC-AUC = 0.8422
RBF SVM ROC-AUC = 0.8082
```

### Probability Quality

```text
XGBoost Brier = 0.1365
RBF SVM Brier = 0.1460
```

Lower Brier score indicates better probabilistic accuracy in this test.

### Efficiency

```text
XGBoost training = 0.593 sec
RBF SVM training = 15.692 sec

XGBoost inference = 0.0438 sec
RBF SVM inference = 0.5607 sec
```

Therefore, for this experiment, XGBoost provided a strong combination of:

```text
Predictive Performance
        +
Probability Quality
        +
Computational Efficiency
```

---

# Production Probability Use Case

The final use case is a hypothetical automated customer-retention system.

The system could use the model's predicted churn probability to identify customers at different risk levels.

For example:

```text
Customer
   ↓
Preprocessing
   ↓
XGBoost Model
   ↓
Predicted Churn Probability
   ↓
Business Retention Rules
   ↓
Retention Offer / Budget Decision
```

However, the model probability should not automatically be treated as a guaranteed real-world probability.

Before deploying a financial decision system, the model should be monitored for:

* Calibration drift
* Data distribution changes
* Churn-rate changes
* Feature drift
* Model performance degradation
* Changes in customer behavior

---

# Key Machine Learning Concepts Learned

## Ensemble Learning

Combining multiple weak or simple learners can produce a stronger model.

## Boosting

Boosting builds models sequentially, with later models correcting previous errors.

## Learning Rate

Controls how strongly each new tree contributes to the model.

## Number of Estimators

Controls how many boosting trees are created.

## Max Depth

Controls the complexity of each decision tree.

## Early Stopping

Stops training when validation performance stops improving.

## SVM

Finds a decision boundary that separates classes while maximizing the margin.

## Kernel Trick

Allows SVM to model nonlinear relationships without explicitly constructing the high-dimensional feature space.

## Calibration

Checks whether predicted probabilities correspond to observed frequencies.

## Brier Score

Measures probabilistic prediction error.

Lower Brier score means better probabilistic accuracy.

## ROC-AUC

Measures how well the model separates and ranks positive and negative examples.

It does **not** directly measure probability calibration.

## F1 Score

Balances precision and recall.

It is especially useful when both false positives and false negatives matter.

---

# Important Data Science Practices Used

### 1. Train/Test Separation

The final test set was kept separate from model training and hyperparameter tuning.

### 2. Validation for Early Stopping

A validation subset was created from the training data for early stopping.

The test set was not used as the early-stopping evaluation set.

### 3. Pipeline-Based Preprocessing

Preprocessing and modeling were combined into pipelines to reduce the risk of inconsistent transformations.

### 4. Unknown Categories

`OneHotEncoder(handle_unknown="ignore")` was used so that unseen categories during transformation do not cause an encoding error.

### 5. Multiple Evaluation Metrics

The models were not evaluated using only one metric.

The project considered:

* F1
* ROC-AUC
* PR-AUC
* Brier Score
* Calibration
* Training time
* Inference time

### 6. Probability Evaluation

Because the intended business use case involves predicted probabilities, calibration and Brier score were included instead of relying only on classification metrics.

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Google Colab
* Jupyter Notebook
* Git
* GitHub

---

# Project Structure

```text
Ensemble_Kernel_Assignment/
│
├── Ensemble_&_Kernel_Model.ipynb
├── Telco-Customer-Churn.csv
└── README.md
```

---

# Conclusion

This project compared ensemble boosting and kernel-based approaches for customer churn prediction.

XGBoost was tuned using cross-validation and evaluated using both classification and probability-based metrics. SVM models were also evaluated, including linear and RBF kernels.

The final XGBoost model achieved:

```text
F1 Score   = 0.5774
ROC-AUC    = 0.8422
Brier Score = 0.1365
```

It also demonstrated lower training and inference time than the RBF SVM in this experiment.

The project demonstrates that model selection should consider more than a single performance metric. For a real-world churn system, **discrimination, probability calibration, computational cost, and data leakage prevention** are all important considerations.
