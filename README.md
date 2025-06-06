# Wine-Quality-Prediction-Using-Logistic-Regression

This project uses a Logistic Regression model to predict the quality of white wine based on its chemical features. The goal is to classify wine samples into different quality categories using supervised machine learning.

📌 Problem Statement
Predict the quality score of white wine based on its physicochemical attributes such as acidity, sugar content, pH, and alcohol.

🧾 Dataset Information
Source: UCI Machine Learning Repository

File Used: winequality-white.csv

Samples: ~4,898

Features: 11 chemical properties

Target: Wine quality score (integer between 3 and 9)

🔧 Methodology
1. Data Loading
Data read from CSV with ; separator.

2. Feature & Label Definition
X: All features excluding 'quality'.

y: Target variable 'quality'.

3. Train-Test Split
80% training, 20% testing using train_test_split().

🤖 Model Used
Model: Logistic Regression (from sklearn.linear_model)

Training: Model trained on X_train, y_train.

Prediction: Quality scores predicted on X_test.

📊 Evaluation
Used accuracy_score and classification_report to evaluate performance.

Bar plot used to visualize the distribution of predicted wine quality scores.

▶️ How to Run
Install required libraries:

pip install pandas numpy seaborn scikit-learn matplotlib
Run the script in Jupyter Notebook or Google Colab.

Make sure the dataset path is correct:

data = pd.read_csv("/content/winequality-white.csv", sep=";")
