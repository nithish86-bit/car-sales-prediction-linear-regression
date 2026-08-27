# Car Sales Prediction using Linear Regression

## 📌 Project Overview

This project uses Linear Regression to predict car sales based on vehicle and pricing-related features.

The analysis follows an end-to-end data science workflow including data understanding, exploratory data analysis (EDA), data cleaning, categorical variable encoding, feature selection, multicollinearity analysis, model building, prediction, and model evaluation.

## 🎯 Objective

To develop a Linear Regression model that predicts car sales and identify the vehicle characteristics that contribute to sales performance.

## 📊 Dataset

The dataset contains information about different car models and their sales performance.

Key variables include:

- Manufacturer
- Model
- Vehicle Type
- Price
- Fuel Efficiency
- Length
- Sales in Thousands

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn
- Statsmodels
- Patsy
- Excel

## 🔎 Analysis Workflow

### 1. Data Understanding & EDA

- Inspected dataset structure and dimensions
- Analyzed categorical and numerical variables
- Checked missing values
- Examined descriptive statistics
- Calculated cardinality of categorical variables
- Analyzed correlations between numerical variables

### 2. Data Cleaning

- Treated extreme values using percentile-based clipping
- Handled missing numerical values using mean imputation
- Converted categorical variables into dummy variables

### 3. Target Variable Transformation

The `Sales_in_thousands` variable was log-transformed to reduce skewness and improve the suitability of the data for regression modeling.

### 4. Feature Selection

Used an F-test to identify statistically significant predictors.

Variables with a p-value below 0.10 were considered for model development.

### 5. Multicollinearity Analysis

Variance Inflation Factor (VIF) was used to identify potential multicollinearity among predictor variables.

### 6. Model Development

The dataset was divided into:

- 70% Training Data
- 30% Testing Data

Multiple Linear Regression models were developed using Statsmodels.

### 7. Model Evaluation

Model performance was evaluated using:

- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)

The model was evaluated on both training and testing datasets.

### 8. Decile Analysis

Decile analysis was performed by ranking predicted sales into ten groups and comparing average actual sales with predicted sales.

### 9. Residual Analysis

Residual analysis was performed to examine model behavior and identify potential limitations of the regression model.

## 📈 Model Features

The final model used the following predictors:

- Fuel Efficiency
- Vehicle Length
- Price in Thousands
- Manufacturer
- Vehicle Type

## 📊 Model Evaluation Metrics

The final model was evaluated using the following metrics:

| Metric | Description |
|---|---|
| MAE | Measures the average absolute prediction error |
| MAPE | Measures prediction error as a percentage |
| MSE | Measures the average squared prediction error |
| RMSE | Measures the square root of the average squared prediction error |

## ⚠️ Model Limitations

The model may have limitations due to:

- Small dataset size
- Regression assumptions
- Possible multicollinearity
- Limited number of predictive variables
- Potential overfitting

## 📁 Repository Structure

```text
car-sales-prediction-linear-regression/
│
├── Car_sales.csv
├── car_sales_prediction.py
├── Linear_correlations.xlsx
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore

---

## 👨‍💻 Author

**Nithish Ramesh**

**GitHub:** [nithish86-bit](https://github.com/nithish86-bit)

**LinkedIn:** [Nithish Ramesh](https://www.linkedin.com/in/nithish-rbn)


