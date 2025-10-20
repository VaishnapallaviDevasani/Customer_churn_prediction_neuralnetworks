# Customer Churn Prediction Using Neural Networks

## Project Overview
This project predicts **customer churn** for a telecom company using a **neural network** built with TensorFlow and Keras.  
Churn prediction is a critical business problem: it helps companies identify customers likely to leave, allowing them to take proactive retention measures.

**Dataset:** [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Project Goal:** Predict whether a customer will churn (leave) based on demographics, account information, services subscribed, and billing data.

---

## Features Used

- **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Account Info:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`
- **Services Subscribed:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Charges:** `MonthlyCharges`, `TotalCharges`
- **Target:** `Churn` → 0 = stay, 1 = leave

---

## Data Preprocessing

1. **Handle Missing Values:** Converted `TotalCharges` to numeric and replaced missing values with 0.
2. **One-Hot Encoding:** Categorical columns (`Yes/No` and text categories) were converted to numeric using one-hot encoding.
3. **Feature Scaling:** Numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`) were standardized using **Z-score normalization** to improve neural network training.
4. **Train-Test Split:** Dataset split into 80% training and 20% testing.

---

## Neural Network Architecture (Week-1 Level)

- **Input Layer:** Number of neurons = number of features
- **Hidden Layer:** 8 neurons with `ReLU` activation
- **Output Layer:** 1 neuron with `Sigmoid` activation (binary classification)

**Loss Function:** Binary Crossentropy  
**Optimizer:** Adam  
**Epochs:** 20-30  
**Batch Size:** 32

---

## Model Evaluation

After training and applying oversampling to balance classes:

| Metric       | Churn=No | Churn=Yes |
|--------------|-----------|-----------|
| Precision    | 0.91      | 0.55      |
| Recall       | 0.76      | 0.80      |
| F1-Score     | 0.83      | 0.65      |
| Support      | 1036      | 373       |

**Overall Accuracy:** 77%  

> Balancing the dataset improved the recall for churners, helping the model identify more customers who are likely to leave.

---

## Improvements Implemented

- **Class Imbalance Handling:** Applied **oversampling** to increase the number of churn cases.
- **Feature Scaling:** Standardized numeric features for better convergence.
- **Threshold Tuning:** Explored prediction thresholds beyond 0.5 to optimize recall.
- **Feature Engineering:** Optionally, new features like `EngagementScore = tenure * MonthlyCharges / (TotalCharges + 1)` can be added for better prediction.

---

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/VaishnapallaviDevasani/Customer_churn_prediction_neuralnetworks.git
