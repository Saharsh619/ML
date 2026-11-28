# Pune House Rent Prediction using Machine Learning

A Machine Learning project that predicts monthly house rent in Pune based on property characteristics such as house size, BHK, furnishing status, number of bathrooms, property type, and location-related information.

The project follows an end-to-end Machine Learning workflow, starting from data understanding and preprocessing to model training and evaluation.

## Project Overview

The objective of this project is to build a regression model that estimates the monthly rental price of residential properties in Pune.

Using a Pune housing dataset, I performed Exploratory Data Analysis (EDA), handled missing values, engineered relevant features, and trained a Linear Regression model to predict house rent.

## Project Structure

```text
ML/
│
└── linear regression/
    │
    ├── data/
    │   ├── Indian_housing_Pune_data.csv
    │   └── cleaned_housing_data.csv
    │
    ├── 01_data_understanding.ipynb
    ├── 02_model_training.ipynb
    │
    └── README.md
```

## Project Workflow

### 1. Data Understanding and Exploratory Data Analysis

Notebook: `01_data_understanding.ipynb`

- Loaded and explored the Pune housing dataset.
- Removed duplicate records.
- Analysed rental price distribution.
- Examined property types and furnishing status.
- Studied the relationship between house size and rental price.
- Used correlation analysis to identify relationships between numerical features.
- Visualized data using Matplotlib and Seaborn.

### 2. Data Cleaning and Preprocessing

- Removed irrelevant columns such as `priceSqFt`, `isNegotiable`, and `description`.
- Handled missing values in `numBathrooms` and `numBalconies` using median imputation.
- Converted house size into numerical values (square feet).
- Converted security deposits into numerical values.
- Removed unnecessary columns such as currency and city, since all records belonged to Pune.
- Converted property furnishing status into numerical features using One-Hot Encoding.

### 3. Feature Engineering

Created additional features to improve the model:

- `house_size_sqft`: Extracted numerical house size in square feet.
- `days_since_posted`: Converted listing age into numerical days.
- `BHK`: Extracted the number of bedrooms.
- `property_type`: Extracted property categories such as Apartment, Villa, and Independent House.
- `location_frequency`: Represented the frequency of properties listed in each location.

After preprocessing, the final dataset contained:

- **3,684 records**
- **18 columns**
- **17 input features**
- **1 target variable (price)**

The cleaned dataset was saved as `data/cleaned_housing_data.csv`.

### 4. Model Training

Notebook: `02_model_training.ipynb`

- Loaded the cleaned dataset.
- Separated independent features (X) and target variable (y).
- Split the dataset into training and testing sets.
- Used an 80:20 train-test split.
- Trained a Linear Regression model using Scikit-learn.

**Model:** Linear Regression

**Train-Test Split:**
- Training samples: 2,947
- Testing samples: 737

## Model Evaluation

The model was evaluated using three regression metrics.

| Evaluation Metric | Result |
|---|---:|
| R² Score | 0.7865 |
| Mean Absolute Error (MAE) | ₹5,422.94 |
| Root Mean Squared Error (RMSE) | ₹9,788.68 |

### Results Interpretation

- The model achieved an R² score of approximately 0.7865, indicating that it explains around 78.65% of the variance in rental prices on the test dataset.
- The Mean Absolute Error indicates an average absolute prediction error of approximately ₹5,423.
- RMSE gives more weight to larger prediction errors and was approximately ₹9,789.

These results provide a baseline for further model improvement.

## Technologies Used

- **Python** – Programming language
- **Pandas** – Data manipulation and preprocessing
- **NumPy** – Numerical operations
- **Matplotlib** – Data visualization
- **Seaborn** – Statistical data visualization
- **Scikit-learn** – Machine Learning and model evaluation
- **Jupyter Notebook** – Development environment

## How to Run the Project

### 1. Clone the Repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the Project Directory

```bash
cd "linear regression"
```

### 3. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 4. Run Jupyter Notebook

```bash
jupyter notebook
```

### 5. Execute the Notebooks in Order

1. Open `01_data_understanding.ipynb` to explore, clean, and preprocess the dataset.
2. Open `02_model_training.ipynb` to train and evaluate the Linear Regression model.

Make sure the original dataset is available at `data/Indian_housing_Pune_data.csv` before running the first notebook.

## Future Improvements

- Experiment with multiple regression algorithms such as Decision Tree, Random Forest, and Gradient Boosting.
- Perform hyperparameter tuning.
- Improve location-based feature representation.
- Analyse outliers and their impact on predictions.
- Compare different models using regression evaluation metrics.
- Deploy the final model using Streamlit.

## Author

**Saharsh Anand**

B.Tech Computer Science Engineering

Interested in Machine Learning, Artificial Intelligence, and Data Science.