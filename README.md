# 🧠 Parkinson's Disease Prediction using Machine Learning 🤖
This project builds a machine learning pipeline to predict Parkinson’s Disease using various classifiers like XGBoost, SVM, and Logistic Regression. The dataset is preprocessed, balanced, scaled, and feature-selected for optimal performance.

# 📁 Files
Parkinson_Disease_Prediction.ipynb: Main Jupyter Notebook containing data preprocessing, modeling, and evaluation code.

parkinson_disease.csv: Dataset containing voice measurements and labels (class) indicating presence or absence of Parkinson’s.

# 🛠️ Technologies Used
Python

pandas, numpy – Data manipulation

matplotlib, seaborn – Visualization 📊

scikit-learn – ML models, preprocessing

xgboost – Boosting classifier

imblearn – For class imbalance handling

tqdm – Progress bars

# 📦 Dataset Overview
Contains biomedical voice measurements.

Target column: class (1: Parkinson's, 0: Healthy)

Multiple observations per patient (id column dropped)

# 🔁 Preprocessing Pipeline
```python
df = pd.read_csv('parkinson_disease.csv')
df.drop('id', axis=1, inplace=True)
df.isnull().sum()  # Checking for missing values
```
# 📊 Exploratory Data Analysis
Correlation matrix between all features

Visual inspection using seaborn

⚖️ Handling Imbalanced Data
```python
X = df.drop('class', axis=1)
y = df['class']

ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_resample(X, y)
```
# 📐 Feature Scaling & Selection
```python
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_resampled)

selector = SelectKBest(score_func=chi2, k=20)
X_selected = selector.fit_transform(X_scaled, y_resampled)
```
# 🤖 Models Used
Support Vector Classifier (SVC)

XGBoost Classifier

Logistic Regression

Each model is trained and evaluated using accuracy and classification metrics.
