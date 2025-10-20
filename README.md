# Customer Churn Prediction using Neural Networks

## 📌 Project Overview
Customer churn is when a customer stops using a company's service. Predicting churn is crucial for telecom companies to **retain customers** and reduce revenue loss.  

This project uses **Week-1 neural network concepts** (Andrew Ng’s Deep Learning Specialization) to predict whether a customer will churn, based on demographic, account, and service usage data.

---

## 🗂 Dataset
The dataset used is the **Telco Customer Churn Dataset**, which contains customer information such as:

- **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Account info:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`
- **Services subscribed:** `PhoneService`, `InternetService`, `OnlineSecurity`, `StreamingTV`, etc.
- **Charges:** `MonthlyCharges`, `TotalCharges`
- **Target variable:** `Churn` (0 = stay, 1 = leave)

**Dataset source:** [Kaggle: Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)

---

## 🧰 Tools & Libraries
- Python 3  
- Pandas, NumPy  
- Scikit-learn  
- TensorFlow & Keras  
- Matplotlib, Seaborn  

---

## ⚙️ Data Preprocessing
1. **Handle missing values:**  
   Convert `TotalCharges` to numeric, fill missing values.  

2. **Convert categorical variables to numeric:**  
   Used **one-hot encoding** for Yes/No and text fields.  

3. **Feature scaling:**  
   Standardized numeric columns (`tenure`, `MonthlyCharges`, `TotalCharges`) using **z-score normalization** for faster neural network training.

4. **Split dataset:**  
   Training set: 80% | Test set: 20%  

5. **Handle imbalanced dataset:**  
   Applied **oversampling** for churners to balance classes.

---

## 🧠 Neural Network Model
- **Architecture:**  
  - Input layer → Hidden layer (8 neurons, ReLU) → Output layer (1 neuron, Sigmoid)
- **Activation functions:**  
  - Hidden layer → ReLU (learns non-linear patterns)  
  - Output layer → Sigmoid (binary classification)
- **Loss function:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Epochs:** 20–30  
- **Batch size:** 32  

---

## 📈 Model Evaluation
- **Final Accuracy:** ~81.26%  
- **Precision, Recall, F1-score:**  
  - Class `False` (Non-churn): High precision → correctly identifies staying customers  
  - Class `True` (Churn): Improved recall after oversampling → better identification of churners  

**Metrics used:**
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

**Tip:** Adjusting prediction threshold can improve recall for churners.

---

## 🔄 Improvements & Feature Engineering
1. **Oversampling/undersampling:** To handle class imbalance.  
2. **Feature engineering:**  
   Example: `EngagementScore = tenure * MonthlyCharges / (TotalCharges + 1)`  
3. **Hidden layer tuning:** Added neurons carefully to avoid overfitting.  
4. **Threshold tuning:** Adjusted prediction threshold for better churn detection.  

---

## 📊 Project Workflow (Flow Chart)

