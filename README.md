# Customer Churn Prediction (Neural Networks)

## Project Overview
This project aims to predict telecom customers who are likely to churn (leave) using a feed-forward neural network. By identifying potential churners early, telecom companies can take targeted actions to retain customers, reducing revenue loss.

The dataset contains 7000+ customers with features such as demographic information, services used, and billing details. The target variable is `Churn` (Yes/No).

---

## Data Preprocessing
- **Feature Cleaning:** Converted `TotalCharges` to numeric and handled missing values.
- **Categorical Variables:** One-hot encoding for categorical columns (gender, contract type, payment method, etc.).
- **Scaling:** StandardScaler applied to numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`) for uniformity.
- **Train-Test Split:** 80% train, 20% test set.

---

## Handling Class Imbalance
The dataset has more non-churners than churners:
- **Before SMOTE:** False: 4138, True: 1496  
- **After SMOTE (on training data only):** False: 4138, True: 4138

**Technique Used:** SMOTE (Synthetic Minority Oversampling Technique) to generate synthetic examples of minority class (churners) and balance the training data.

---

## Neural Network Model
- **Architecture:**
  - Input Layer → Dense(32, ReLU) → Dropout(0.3)
  - Dense(16, ReLU) → Dropout(0.3)
  - Output Layer: Dense(1, Sigmoid)
- **Activation Functions:**  
  - ReLU for hidden layers  
  - Sigmoid for binary classification output
- **Regularization:** L2 regularization and Dropout used to prevent overfitting
- **Optimizer:** Adam  
- **Loss Function:** Binary Crossentropy
- **Threshold for predicting churn:** 0.4 (optimized for better recall of churners)

---
## Before Changes

## Model Evaluation

After training and applying oversampling to balance classes:

| Metric       | Churn=No | Churn=Yes |
|--------------|-----------|-----------|
| Precision    | 0.91      | 0.55      |
| Recall       | 0.76      | 0.80      |
| F1-Score     | 0.83      | 0.65      |
| Support      | 1036      | 373       |

**Overall Accuracy:** 77%  

## Results on Test Set
| Metric | Non-Churn (False) | Churn (True) | Overall |
|--------|-----------------|---------------|---------|
| Precision | 0.88 | 0.62 | - |
| Recall    | 0.85 | 0.70 | - |
| F1-score  | 0.86 | 0.66 | - |
| Accuracy  | - | - | 81.33% |

**Interpretation:**  
- Recall for churners improved from 0.56 → 0.70 compared to baseline Week-1 model.  
- Overall accuracy remains high (~81%).  
- Model now detects 70% of potential churners, enabling targeted retention efforts.

---

## Key Improvements from Week-1
1. **SMOTE oversampling:** Balanced training data for better learning of minority class.
2. **Threshold tuning:** Lowered decision threshold to 0.4 to improve churner recall.
3. **Network capacity increase:** More neurons in hidden layers (32 → 16) for richer learning.
4. **Regularization:** Added Dropout and L2 to prevent overfitting on synthetic samples.

---

## Technologies Used
- Python, Pandas, NumPy, Scikit-learn
- TensorFlow, Keras
- Matplotlib, Seaborn
- Imbalanced-learn (SMOTE)

---

