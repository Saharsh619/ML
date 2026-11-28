# Customer Purchase Prediction using Logistic Regression

## Project Overview

This is a Machine Learning mini-project that predicts whether a customer will purchase a product based on their Age and Estimated Salary.

The project uses Logistic Regression, a supervised Machine Learning classification algorithm, to classify customers into two categories:

- 0 – Customer will not purchase the product.
- 1 – Customer will purchase the product.

The main objective is to understand the complete Machine Learning workflow, from data understanding and preprocessing to model training, evaluation, and prediction.

## Dataset

**Dataset Name:** Social Network Ads

The dataset contains 400 customer records and 5 columns.

| Column | Description |
|---|---|
| User ID | Unique customer identifier |
| Gender | Customer gender |
| Age | Customer age |
| EstimatedSalary | Customer estimated salary |
| Purchased | Target variable (0 or 1) |

### Target Variable

- 0 = Not Purchased
- 1 = Purchased

## Project Structure

```text
Logistic Regression/
│
├── data/
│   ├── Social_Network_Ads.csv
│   └── cleaned_social_network_ads.csv
│
├── models/
│
├── 01_data_understanding.ipynb
├── 02_model_training.ipynb
└── README.md
```

## 1. Data Understanding and Exploratory Data Analysis

Notebook: `01_data_understanding.ipynb`

The first step was understanding the dataset using Pandas, Matplotlib, and Seaborn.

The following operations were performed:

- Loaded and inspected the dataset.
- Checked the number of rows and columns.
- Examined data types using `info()`.
- Analysed numerical features using `describe()`.
- Checked missing values.
- Checked duplicate records.
- Analysed the distribution of purchasing and non-purchasing customers.
- Visualized relationships between Age, Estimated Salary, and Purchased.

### Purchase Distribution

| Purchased | Number of Customers |
|---|---:|
| 0 (Not Purchased) | 257 |
| 1 (Purchased) | 143 |

## 2. Data Cleaning and Preprocessing

The dataset was already relatively clean, with no missing values or duplicate records.

The following preprocessing steps were performed:

- Removed `User ID` because it does not provide useful predictive information.
- Removed `Gender` for the initial experiment.
- Selected `Age` and `EstimatedSalary` as independent variables.
- Selected `Purchased` as the target variable.

The cleaned dataset contains:

- 400 rows
- 3 columns

The cleaned dataset was saved as:

`data/cleaned_social_network_ads.csv`

The original dataset was kept unchanged.

## 3. Feature Selection

The independent and dependent variables were separated.

**Independent Variables (X):**
- Age
- EstimatedSalary

**Dependent Variable (y):**
- Purchased

```python
X = df.drop(columns=['Purchased'])
y = df['Purchased']
```

## 4. Train-Test Split

The dataset was divided into training and testing sets using Scikit-learn's `train_test_split()`.

Parameters:

- Training data: 80%
- Testing data: 20%
- Random state: 42
- Stratification: Enabled

| Dataset | Records |
|---|---:|
| Training | 320 |
| Testing | 80 |

Stratification was used to maintain the proportion of both classes in the training and testing datasets.

## 5. Feature Scaling

StandardScaler was used to standardize the independent variables.

The reason for scaling was that Age and EstimatedSalary have different numerical ranges.

The standardization formula is:

z = (x - mean) / standard deviation

The scaler was fitted only on the training data and then applied to the testing data to prevent data leakage.

## 6. Model Training

**Algorithm Used:** Logistic Regression

The Logistic Regression model was implemented using Scikit-learn.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)
```

The model learns the relationship between customer Age, Estimated Salary, and purchase behaviour.

## 7. Model Evaluation

The trained model was evaluated using the testing dataset.

The following evaluation metrics were used:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC Score

### Model Performance

| Metric | Result |
|---|---:|
| Accuracy | 83.75% |
| ROC-AUC | 0.9053 |

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 | 0.83 | 0.94 | 0.88 | 51 |
| 1 | 0.86 | 0.66 | 0.75 | 29 |
| Macro Average | 0.85 | 0.80 | 0.81 | 80 |
| Weighted Average | 0.84 | 0.84 | 0.83 | 80 |

### Interpretation

- The model achieved an accuracy of 83.75%.
- It correctly classified 67 out of 80 customers.
- The recall for class 0 was 94%.
- The recall for class 1 was 66%.
- The ROC-AUC score of 0.9053 indicates good classification performance.

The model performs better at identifying customers who do not purchase than customers who purchase.

## 8. Threshold Experimentation

Logistic Regression uses a default classification threshold of 0.5.

Different thresholds were tested to understand their effect on model performance.

| Threshold | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| 0.3 | 78.75% | 65.79% | 86.21% | 74.63% |
| 0.4 | 80.00% | 70.97% | 75.86% | 73.33% |
| 0.5 | 83.75% | 86.36% | 65.52% | 74.51% |
| 0.6 | 77.50% | 82.35% | 48.28% | 60.87% |
| 0.7 | Not recorded | 92.86% | 44.83% | 60.47% |

### Observations

- Lower thresholds increase recall but can reduce precision.
- Higher thresholds increase precision but can reduce recall.
- A threshold of 0.3 achieved the highest recall among the tested thresholds.
- A threshold of 0.5 achieved the highest accuracy among the tested thresholds.

Threshold selection depends on the business objective.

## 9. Decision Boundary Visualization

A decision boundary was plotted using Age and EstimatedSalary.

The visualization shows how Logistic Regression separates customers into two classes.

- Pink region represents predicted class 0 (Not Purchased).
- Green region represents predicted class 1 (Purchased).
- Red points represent actual class 0.
- Green points represent actual class 1.

The visualization helps understand how the model classifies customers based on their age and estimated salary.

## 10. Model Coefficient Interpretation

The trained model produced the following coefficients:

| Parameter | Value |
|---|---:|
| Intercept | -1.12123469 |
| Age | 2.18068711 |
| EstimatedSalary | 1.24342738 |

The Logistic Regression equation is:

z = -1.1212 + 2.1807(Age) + 1.2434(EstimatedSalary)

Here, Age and EstimatedSalary represent standardized feature values.

### Observations

- Both coefficients are positive.
- Age has a larger positive coefficient than EstimatedSalary.
- Increasing either standardized feature increases the predicted log-odds of purchasing, keeping the other feature constant.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## How to Run the Project

### 1. Clone the Repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the Project Directory

```bash
cd "Logistic Regression"
```

### 3. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter joblib
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

### 5. Run Notebooks in Order

1. `01_data_understanding.ipynb`
2. `02_model_training.ipynb`

Make sure the original dataset is available inside the `data` folder.

## Future Improvements

- Save the trained model using Joblib.
- Predict purchases for new customers.
- Experiment with additional customer features.
- Perform cross-validation.
- Compare Logistic Regression with other classification algorithms.
- Build an interactive prediction application using Streamlit.

## Author

**Saharsh Anand**

B.Tech Computer Science Engineering

Interested in Machine Learning, Artificial Intelligence, and Data Science.