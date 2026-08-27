import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error
)
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split

from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf


# ============================================================
# CAR SALES PREDICTION - LINEAR REGRESSION
# ============================================================


# ------------------------------------------------------------
# 1. DATA LOADING
# ------------------------------------------------------------

cars = pd.read_csv("Car_sales.csv")

print(cars.head(3))


# ------------------------------------------------------------
# 2. DATA UNDERSTANDING / EDA
# ------------------------------------------------------------

cars.info()

print("No. of observations in dataset =", cars.shape[0])
print("No. of variables in dataset =", cars.shape[1])


# Fetching categorical data
print(cars.loc[:, ["Manufacturer", "Model", "Vehicle_type"]])


# Cardinality of categorical variables
print("Manufacturer cardinality =", cars.Manufacturer.nunique())
print(cars.Manufacturer.value_counts())

print("Model cardinality =", cars.Model.nunique())
print("Vehicle Type cardinality =", cars.Vehicle_type.nunique())


# Descriptive analysis of continuous variables
print(cars.describe().T)


# Check for missing data
print(cars.isna().sum())
print(cars.isna().sum() / cars.shape[0] * 100)


# ------------------------------------------------------------
# 3. CORRELATION ANALYSIS
# ------------------------------------------------------------

Continuous_Var_Relationships = cars.select_dtypes(
    include="number"
).corr()

print(Continuous_Var_Relationships)

plt.figure(figsize=(10, 10))
sns.heatmap(Continuous_Var_Relationships, annot=True)
plt.title("Correlation Matrix")
plt.show()


# Separate categorical and numerical variables
cars_object = cars.select_dtypes(include="object")
cars_numeric = cars.select_dtypes(include="number")

print(cars_object.head(3))
print(cars_numeric.head(3))


# ------------------------------------------------------------
# 4. CONTINUOUS VARIABLE SUMMARY
# ------------------------------------------------------------

def continuous_var_summary(x):

    n_total = x.shape[0]
    n_miss = x.isna().sum()
    per_miss = n_miss * 100 / n_total

    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)

    iqr = q3 - q1

    lc_iqr = q1 - 1.5 * iqr
    uc_iqr = q3 + 1.5 * iqr

    return pd.Series(
        [
            x.dtypes,
            x.nunique(),
            n_total,
            x.count(),
            n_miss,
            per_miss,
            x.sum(),
            x.mean(),
            x.std(),
            x.var(),
            lc_iqr,
            uc_iqr,
            x.min(),
            x.quantile(0.01),
            x.quantile(0.05),
            x.quantile(0.10),
            x.quantile(0.25),
            x.quantile(0.50),
            x.quantile(0.75),
            x.quantile(0.90),
            x.quantile(0.95),
            x.quantile(0.99),
            x.max()
        ],
        index=[
            "dtype",
            "cardinality",
            "n_tot",
            "n",
            "n_miss",
            "p_miss",
            "sum",
            "mean",
            "std",
            "var",
            "lc_iqr",
            "uc_iqr",
            "min",
            "p1",
            "p5",
            "p10",
            "p25",
            "p50",
            "p75",
            "p90",
            "p95",
            "p99",
            "max"
        ]
    )


print(cars_numeric.apply(continuous_var_summary))


# ------------------------------------------------------------
# 5. CATEGORICAL VARIABLE ANALYSIS
# ------------------------------------------------------------

print(cars_object.describe())


# ------------------------------------------------------------
# 6. DATA CLEANING
# ------------------------------------------------------------

# Outlier treatment
cars_numeric = cars_numeric.apply(
    lambda x: x.clip(
        lower=x.quantile(0.01),
        upper=x.quantile(0.99)
    )
)


# Missing value treatment
cars_numeric = cars_numeric.apply(
    lambda x: x.fillna(x.mean())
)


print(cars_numeric.apply(continuous_var_summary))


# ------------------------------------------------------------
# 7. CATEGORICAL VARIABLE ENCODING
# ------------------------------------------------------------

cars_object = pd.get_dummies(
    cars_object.loc[:, ["Manufacturer", "Vehicle_type"]],
    drop_first=True
)

cars_object.rename(
    columns={
        "Manufacturer_Mercedes-B": "Manufacturer_Mercedes_B"
    },
    inplace=True
)

print(cars_object.columns)


# ------------------------------------------------------------
# 8. COMBINE NUMERICAL & CATEGORICAL DATA
# ------------------------------------------------------------

cars_new = pd.concat(
    [cars_numeric, cars_object],
    axis=1
)

print(cars_new.head())


# ------------------------------------------------------------
# 9. TARGET VARIABLE TRANSFORMATION
# ------------------------------------------------------------

sns.histplot(
    cars_new.Sales_in_thousands,
    kde=True
)
plt.title("Distribution of Car Sales")
plt.show()

print(
    "Skewness before log transformation =",
    cars_new.Sales_in_thousands.skew()
)


cars_new["lg_sales_in_thousands"] = np.log(
    cars_new.Sales_in_thousands
)

print(
    "Skewness after log transformation =",
    cars_new["lg_sales_in_thousands"].skew()
)


sns.histplot(
    cars_new["lg_sales_in_thousands"],
    kde=True
)
plt.title("Log-transformed Sales Distribution")
plt.show()


# ------------------------------------------------------------
# 10. CORRELATION MATRIX
# ------------------------------------------------------------

Linear_corelations = cars_new.corr()

Linear_corelations.to_excel(
    "Linear_correlations.xlsx",
    index=False
)


# ------------------------------------------------------------
# 11. FEATURE SELECTION USING F-TEST
# ------------------------------------------------------------

features = cars_new.columns.difference(
    ["Sales_in_thousands", "lg_sales_in_thousands"]
)

print("No. of X variables =", len(features))
print("No. of observations =", cars_new.shape[0])
print(list(features))


f_score, p_values = f_regression(
    cars_new[features],
    cars_new.lg_sales_in_thousands
)


significant_variables = pd.DataFrame()

significant_variables["Features"] = features
significant_variables["F_score"] = f_score
significant_variables["P_values"] = p_values

print(significant_variables)


# Keep variables with p-value < 0.1
features = list(
    significant_variables.loc[
        significant_variables.P_values < 0.1,
        "Features"
    ].reset_index(drop=True)
)

print("Significant features:")
print(features)


# ------------------------------------------------------------
# 12. MULTICOLLINEARITY CHECK USING VIF
# ------------------------------------------------------------

model_param = (
    "lg_sales_in_thousands ~ " +
    "+".join(features)
)

print(model_param)


y, x = dmatrices(
    model_param,
    cars_new,
    return_type="dataframe"
)


vif = pd.DataFrame()

vif["Features"] = x.columns

vif["VIF factor"] = [
    variance_inflation_factor(
        x.values,
        i
    )
    for i in range(x.shape[1])
]

print(vif)


# ------------------------------------------------------------
# 13. TRAIN / TEST SPLIT
# ------------------------------------------------------------

train, test = train_test_split(
    cars_new,
    train_size=0.7,
    test_size=0.3,
    random_state=10
)

print(train.head())
print(test.head())


# ------------------------------------------------------------
# 14. INITIAL LINEAR REGRESSION MODEL
# ------------------------------------------------------------

x = "+".join(features)

formula_0 = (
    "lg_sales_in_thousands ~ " +
    x
)

print(formula_0)


m0 = smf.ols(
    formula_0,
    data=train
).fit()

print(m0.summary())


# ------------------------------------------------------------
# 15. FINAL MODEL AFTER VARIABLE REDUCTION
# ------------------------------------------------------------

formula = """
lg_sales_in_thousands ~
Fuel_efficiency +
Manufacturer_Ford +
Price_in_thousands +
Vehicle_type_Passenger +
Length
"""

lm = smf.ols(
    formula,
    data=train
).fit()

print(lm.summary())


# ------------------------------------------------------------
# 16. MODEL 1
# ------------------------------------------------------------

formula_1 = (
    "lg_sales_in_thousands ~ " +
    " + ".join(features)
)

m1 = smf.ols(
    formula_1,
    data=train
).fit()

print(m1.summary())


# ------------------------------------------------------------
# 17. MODEL 2 - FINAL MODEL
# ------------------------------------------------------------

formula_2 = (
    "lg_sales_in_thousands ~ "
    "Fuel_efficiency + "
    "Length + "
    "Manufacturer_Plymouth + "
    "Price_in_thousands + "
    "Vehicle_type_Passenger"
)

m2 = smf.ols(
    formula_2,
    data=train
).fit()

print(m2.summary())


# ------------------------------------------------------------
# 18. PREDICTION
# ------------------------------------------------------------

train["y_predict"] = np.exp(
    m2.predict(train)
)

test["y_predict"] = np.exp(
    m2.predict(test)
)


print(
    train[
        ["Sales_in_thousands", "y_predict"]
    ].head(4)
)

print(
    test[
        ["Sales_in_thousands", "y_predict"]
    ].head(4)
)


# ------------------------------------------------------------
# 19. MODEL EVALUATION
# ------------------------------------------------------------

train_mae = mean_absolute_error(
    train.Sales_in_thousands,
    train.y_predict
)

test_mae = mean_absolute_error(
    test.Sales_in_thousands,
    test.y_predict
)


train_mape = mean_absolute_percentage_error(
    train.Sales_in_thousands,
    train.y_predict
)

test_mape = mean_absolute_percentage_error(
    test.Sales_in_thousands,
    test.y_predict
)


train_mse = mean_squared_error(
    train.Sales_in_thousands,
    train.y_predict
)

test_mse = mean_squared_error(
    test.Sales_in_thousands,
    test.y_predict
)


train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)


print(
    "MAE for train data =",
    train_mae,
    "| MAE for test data =",
    test_mae
)

print(
    "MAPE for train data =",
    train_mape,
    "| MAPE for test data =",
    test_mape
)

print(
    "MSE for train data =",
    train_mse,
    "| MSE for test data =",
    test_mse
)

print(
    "RMSE for train data =",
    train_rmse,
    "| RMSE for test data =",
    test_rmse
)


# ------------------------------------------------------------
# 20. ACTUAL VS PREDICTED VALUES
# ------------------------------------------------------------

sns.scatterplot(
    x=train.Sales_in_thousands,
    y=train.y_predict
)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()


# ------------------------------------------------------------
# 21. DECILE ANALYSIS
# ------------------------------------------------------------

train["Deciles"] = pd.qcut(
    train.y_predict,
    10,
    labels=range(1, 11)
)

test["Deciles"] = pd.qcut(
    test.y_predict,
    10,
    labels=range(1, 11)
)


train_deciles = (
    train.groupby("Deciles", observed=False)[
        ["Sales_in_thousands", "y_predict"]
    ]
    .mean()
    .reset_index()
)

test_deciles = (
    test.groupby("Deciles", observed=False)[
        ["Sales_in_thousands", "y_predict"]
    ]
    .mean()
    .reset_index()
)


print("Train Decile Analysis:")
print(train_deciles)

print("Test Decile Analysis:")
print(test_deciles)


# ------------------------------------------------------------
# 22. RESIDUAL ANALYSIS
# ------------------------------------------------------------

print("Mean residual =", m2.resid.mean())

print(
    "Pearson correlation:",
    stats.pearsonr(
        m2.resid,
        train.lg_sales_in_thousands
    )
)


sns.scatterplot(
    x=m2.resid,
    y=train.lg_sales_in_thousands
)

plt.xlabel("Residuals")
plt.ylabel("Log Sales")
plt.title("Residual Analysis")
plt.show()


# ------------------------------------------------------------
# 23. MODEL LIMITATIONS
# ------------------------------------------------------------

# Possible reasons for poor model accuracy:
#
# 1. Small sample size
# 2. Regression assumptions may not be completely satisfied
# 3. Possible overfitting
# 4. Data preparation issues
# 5. Important variables may not be included
# 6. Multicollinearity
# 7. Limited explanatory power of available variables
