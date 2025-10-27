# ⚡️ Energy Demand Forecasting Pipeline

This project implements a **modular energy demand forecasting system** using classical machine learning models such as **Random Forest**, **XGBoost**, **Gradient Boosting (Quantile Regression)**, and **Neural Networks (MLP)**.  
It supports full data preprocessing, feature engineering, Bayesian hyperparameter optimization, time series cross-validation, and model evaluation.

---
## 📁 Project Structure
```text
energy_forecasting_project/
│
├── config/
│   ├── config.py                     # Config variables and hyperparameter search spaces
│
├── data_pipeline/
│   ├── pipeline.py                   # EnergyDemandForecasting class
│
├── data/
│   ├── FEATURE_ENGINEERED_DATA.csv   # Example processed dataset
│
├── logs/
│   ├── pipeline.log                  # Training and tuning logs
│
├── notebooks/
│   ├── model_training.ipynb          # Notebook for interactive experiments
│
└── README.md                         # Project documentation

```
---

## 🧠 Overview

The pipeline supports **end-to-end forecasting**, including:

1. **Data Loading** – Load feature-engineered dataset  
2. **Preprocessing** – Generate lag features, rolling statistics, and percentage changes  
3. **Dataset Splitting** – Chronological train-test split for time series  
4. **Model Training** – Bayesian and Grid Search optimization for multiple model types  
5. **Quantile Regression (QRF)** – Predict multiple quantiles (e.g., 10th, 50th, 90th percentile)  
6. **Model Evaluation** – Compute MAE and RMSE  
7. **Logging** – Full training log saved to `logs/pipeline.log`

---

## ⚙️ Class: `EnergyDemandForecasting`

### **1️⃣ Initialization**

```python
from data_pipeline.pipeline import EnergyDemandForecasting
ed = EnergyDemandForecasting(target='load_da')
```

### 2️⃣ Data Loading

```python
df = ed.data_loading(FEATURE_ENGINEERED_DATA_PATH)
```

### 3️⃣ Data Preprocessing

```python
df_processed = ed.preprocessing(df, inference=False)
```
What it does:
- Creates shifted targets: load_da, load_d2
- Generates lag features: lag_1, lag_2, lag_3
- Computes percentage changes for temperature and power
- Adds rolling mean and std windows
- Drops irrelevant columns
- Saves inference-ready data if inference=True

### 4️⃣ Dataset Splitting
```python
X_train, X_test, y_train, y_test = ed.get_model_split_training_datasets(df_processed)
```
- Splitting Logic:
- Chronological split (not random)

### 5️⃣ Model Training
#### a) Random Forest
rf_mod = ed.model_training(X_train=X_train, y_train=y_train, model_name='rf')

#### b) XGBoost
xgb_mod = ed.model_training(X_train=X_train, y_train=y_train, model_name='xgb')

#### c) Meta-Learning (MLP)
meta_mod = ed.model_training(X_train=X_train, y_train=y_train, model_name='meta')

#### d) Quantile Regression (QRF Equivalent)
qrf_models = ed.model_training(X_train=X_train, y_train=y_train, model_name='qrf')

Each quantile model (0.1, 0.5, 0.9) is trained separately using the same optimized hyperparameters.

### 6️⃣ Model Prediction
y_pred = rf_mod.predict(X_test)

For quantile regression:
preds = {q: model.predict(X_test) for q, model in qrf_models.items()}
pred_df = pd.DataFrame(preds)
pred_df.columns = [f"q{int(q*100)}" for q in ed.quantiles]

### 7️⃣ Model Evaluation
mae, rmse = ed.model_eval(y_test, y_pred)

Metrics Returned:
- MAE — Mean Absolute Error
- RMSE — Root Mean Squared Error

## 🧩 Model Overview

This project integrates multiple regression models, each with unique strengths for **energy demand forecasting** tasks.  
Below is a brief overview of the models implemented in the pipeline.

---

### 🌲 1. Random Forest Regressor (`rf`)

**Type:** Ensemble (Bagging-based) Decision Tree Regressor  
**Library:** `scikit-learn`  

Random Forest is an ensemble method that builds multiple decision trees on random subsets of data and features, and averages their predictions.  
It reduces variance compared to a single decision tree and performs well with **nonlinear relationships** and **mixed feature types**.

**Advantages:**
- Handles nonlinear and complex interactions well  
- Robust to noise and overfitting (due to averaging)  
- Requires minimal preprocessing (no scaling or normalization)

**Use in this project:**
- Acts as a **baseline model** for predicting energy demand  
- Tuned using **Bayesian Optimization** to find optimal depth, estimators, and splits

---

### ⚙️ 2. XGBoost Regressor (`xgb`)

**Type:** Gradient Boosting Decision Trees  
**Library:** `xgboost`  

XGBoost (Extreme Gradient Boosting) builds trees sequentially — each new tree corrects the residuals (errors) of the previous ones.  
It’s highly optimized for speed and performance, making it ideal for structured/tabular data.

**Advantages:**
- Excellent predictive accuracy on structured data  
- Supports regularization (`lambda`, `alpha`) to reduce overfitting  
- Handles missing values internally  
- Efficient with large datasets

**Use in this project:**
- Trained with **Bayesian Optimization** to fine-tune learning rate, max depth, and tree complexity  
- Often used for **short-term load forecasting** due to its high accuracy and stability

---

### 🧠 3. MLP Regressor (`meta`)

**Type:** Neural Network (Feedforward Multi-Layer Perceptron)  
**Library:** `scikit-learn`  

The MLP Regressor is a fully connected neural network that models complex, nonlinear relationships between inputs and outputs.  
It’s used here as a **meta-learning model** that can capture deeper relationships not easily modeled by tree-based methods.

**Advantages:**
- Captures nonlinear patterns  
- Learns complex feature interactions  
- Can generalize well when tuned properly  

**Use in this project:**
- Used as a **meta-learner** (stacking layer) on top of base regressors  
- Tuned via **Grid Search** over hidden layer sizes, activation functions, and learning rates

---

### 📈 4. Quantile Gradient Boosting Regressor (`qrf`)

**Type:** Gradient Boosting Regressor with Quantile Loss  
**Library:** `scikit-learn`  

The Quantile Regression version of Gradient Boosting estimates conditional quantiles instead of the mean.  
Instead of predicting a single expected value, it predicts intervals such as the **10th, 50th, and 90th percentiles**, giving insights into prediction uncertainty.

**Advantages:**
- Provides **prediction intervals (confidence bands)**  
- Robust to outliers  
- Useful for **risk-aware forecasting**

**Use in this project:**
- Median (0.5 quantile) model tuned using Bayesian Optimization  
- Separate models trained for each quantile (0.1, 0.5, 0.9)  
- Helps quantify uncertainty in **energy demand forecasts**

---

## 🧮 Model Comparison Summary

| Model | Algorithm Type | Key Feature | Optimization | Output Type |
|--------|----------------|-------------|---------------|--------------|
| **Random Forest** | Bagging Ensemble | Reduces variance via averaging | Bayesian Search | Point estimate |
| **XGBoost** | Boosting Ensemble | Corrects residuals sequentially | Bayesian Search | Point estimate |
| **MLP Regressor** | Neural Network | Learns complex nonlinear mappings | Grid Search | Point estimate |
| **Quantile GB Regressor (QRF)** | Boosting (Quantile Loss) | Estimates prediction intervals | Bayesian + Multi-fit | Quantile estimates |

---

### 💡 Practical Insight

- Use **Random Forest** for interpretability and quick baselines.  
- Use **XGBoost** for best accuracy and robustness to noisy data.  
- Use **MLP** when feature interactions are highly nonlinear.  
- Use **Quantile GB Regressor** when uncertainty intervals are required — e.g., predicting **lower, median, and upper bounds** of energy demand.

## 🎯 Hyperparameter Optimization Techniques

This project employs two different hyperparameter optimization strategies depending on the model type and computational trade-offs:

---

### 🔍 1. Bayesian Optimization (via `BayesSearchCV`)

**Used for:**  
- `RandomForestRegressor` (`rf`)  
- `XGBRegressor` (`xgb`)  
- `GradientBoostingRegressor (Quantile)` (`qrf`)

**Library:** `scikit-optimize (skopt)`

**Concept:**
Bayesian Optimization builds a **probabilistic model** (usually Gaussian Process or Tree-structured Parzen Estimator) of the objective function to efficiently explore the hyperparameter space.  
It balances **exploration (trying new hyperparameters)** and **exploitation (focusing on promising regions)**.

Unlike random or grid search, Bayesian optimization *learns* from previous evaluations to propose smarter hyperparameter configurations in each iteration.

**Advantages:**
- Finds near-optimal parameters with fewer iterations  
- Efficient for expensive models (like XGBoost or GBR)  
- Handles continuous and integer search spaces


## ⚙️ Grid Search Optimization

**Used For:**  
- `MLPRegressor` (Meta-Learning Model)

**Library:**  
- `scikit-learn`

---

### 🧩 What is Grid Search?

Grid Search is a **systematic and exhaustive hyperparameter tuning method**.  
It evaluates **all possible combinations** of hyperparameter values that you define in a grid and identifies the configuration that yields the **best model performance** according to a chosen metric (e.g., RMSE, MAE).

It’s particularly useful when:
- You have a **small or moderate** number of hyperparameters.
- You want to **guarantee** testing every possible combination.
- You want **deterministic, reproducible results**.

---

### 🧠 How It Works

1. Define a **parameter grid** — a dictionary of hyperparameters with possible values.
2. For every combination of parameters:
   - Train the model on the training folds.
   - Evaluate it on the validation fold.
3. Select the combination that gives the best cross-validation score.

---
