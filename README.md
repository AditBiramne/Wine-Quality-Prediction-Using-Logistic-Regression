# 🍷 Wine Quality Prediction Using Logistic Regression

This project applies **Logistic Regression** to predict the **quality of white wine** based on its chemical features. The goal is to classify wine samples into quality categories using **supervised machine learning** on a real-world dataset.

It serves as a beginner-friendly project in classification tasks, data preprocessing, and model evaluation.

---

📌 Problem Statement

- Predict the **quality score of white wine** (ranging from 3 to 9) based on physicochemical attributes like:
  - Fixed acidity
  - Volatile acidity
  - Citric acid
  - Residual sugar
  - Chlorides
  - Free sulfur dioxide
  - Total sulfur dioxide
  - Density
  - pH
  - Sulphates
  - Alcohol

---

🧾 Dataset Information

- **Source**: UCI Machine Learning Repository  
- **File Used**: `winequality-white.csv`
- **Samples**: ~4,898 white wine records  
- **Features**: 11 numerical chemical properties  
- **Target**: Quality score (`int`, from 3 to 9)

---

🔧 Methodology

### 📥 Data Loading

- Read CSV with semicolon separator:

```python
import pandas as pd

data = pd.read_csv('/content/winequality-white.csv', sep=';')
```

### 🎯 Feature & Label Definition

```python
X = data.drop('quality', axis=1)
y = data['quality']
```

### 🔀 Train-Test Split

- 80% training, 20% testing:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

🤖 Model: Logistic Regression

- Implemented using `sklearn.linear_model.LogisticRegression`

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

📊 Model Evaluation

- Metrics Used:
  - `accuracy_score`
  - `classification_report`
  - `confusion_matrix`

```python
from sklearn.metrics import accuracy_score, classification_report

print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))
```

- Optional: Visualize predicted vs actual distribution:

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x=predictions)
plt.title("Predicted Wine Quality Distribution")
plt.xlabel("Quality Score")
plt.ylabel("Count")
plt.show()
```

---

▶️ How to Run

1. ✅ Install required libraries:

```bash
pip install pandas numpy seaborn scikit-learn matplotlib
```

2. ✅ Run the notebook or script in:
   - Google Colab
   - Jupyter Notebook

3. ✅ Make sure dataset path is correct:

```python
data = pd.read_csv("/content/winequality-white.csv", sep=";")
```

---

🚀 Project Highlights

- Demonstrates how logistic regression can be adapted for **multi-class classification**
- Shows the impact of **imbalanced classes** on prediction quality
- Useful for **introductory ML learning** and exploring **evaluation metrics**
- Offers a practical case of **feature selection** and preprocessing in a real dataset

---

🛠 Future Enhancements

- Try **Random Forests** or **Gradient Boosting** for improved accuracy
- Apply **SMOTE** to balance the dataset
- Perform **feature importance analysis**
- Add **cross-validation** for robust evaluation
- Deploy using Flask or Streamlit for web-based prediction

---

📚 References

- UCI Wine Dataset: https://archive.ics.uci.edu/ml/datasets/Wine+Quality
- scikit-learn Documentation: https://scikit-learn.org/stable/

