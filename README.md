# Bias in Machine Learning Models: Detection and Reduction

This mini-project demonstrates how to **detect and reduce bias** in machine learning models using the **UCI Adult Income Dataset**.  
We evaluate fairness metrics and apply a **Reweighing** technique to mitigate bias against disadvantaged groups.

---

## 📌 Project Overview

Machine learning models often inherit societal biases from the data they are trained on.  
In this project, we:  
- Train a **Logistic Regression** model on income prediction.  
- Measure fairness using **AIF360 metrics**:
  - Statistical Parity Difference
  - Equal Opportunity Difference
  - Average Odds Difference  
- Apply **Reweighing** to adjust sample weights and reduce bias.  
- Compare fairness metrics before and after bias mitigation.

---

## 🛠️ Tech Stack

- **Python 3.x**
- **Pandas, NumPy, Matplotlib** – Data analysis and visualization
- **Scikit-learn** – Model training
- **AIF360** – Fairness metrics & bias mitigation

---

## 📂 Project Structure
# Bias in Machine Learning Models: Detection and Reduction

This mini-project explores how bias in machine learning models can be detected and reduced.  
We use the **UCI Adult Income dataset**, where the goal is to predict whether a person earns more than \$50K annually.  
However, models trained on this data often exhibit **bias against certain demographic groups**, particularly based on **gender** and **race**.

We demonstrate:  
1. Training a baseline logistic regression model.  
2. Measuring fairness using **AIF360 metrics**.  
3. Applying **Reweighing** to mitigate bias.  
4. Comparing fairness metrics before and after mitigation.  

**Key Libraries**: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `aif360`
!pip install pandas numpy matplotlib scikit-learn aif360
## 1. Load and Explore Dataset

We use the UCI Adult Income dataset, which contains demographic data and income labels.  
The target variable is **income > \$50K (1)** or **≤ \$50K (0)**.
## 2. Train-Test Split

We split the dataset into **70% training** and **30% testing**.
## 3. Data Preprocessing

We scale numeric features and one-hot encode categorical features using a `ColumnTransformer`.
## 4. Baseline Model Training

We use a **Logistic Regression** classifier without bias mitigation.
## 5. Convert to AIF360 Dataset

We convert our test set into an AIF360 `BinaryLabelDataset` object for fairness evaluation.  
The **protected attribute** is `sex` (Male vs. Female).
## 6. Fairness Metrics for Baseline Model

We compute key fairness metrics:
- **Statistical Parity Difference**
- **Equal Opportunity Difference**
- **Average Odds Difference**
## 7. Bias Mitigation with Reweighing

We reweight training samples to give **disadvantaged groups more weight**.
## 8. Comparing Fairness Metrics

We compare fairness metrics **before vs. after mitigation** using a bar chart.
## 9. Conclusion

- The baseline model exhibited measurable **bias** against certain groups (e.g., females).  
- By applying **Reweighing**, we successfully reduced bias metrics (closer to 0 is fairer).  
- This demonstrates how **pre-processing bias mitigation techniques** can improve fairness.  

Future improvements:  
- Explore **post-processing methods** like Reject Option Classification.  
- Experiment with **adversarial debiasing** to learn fair representations.
