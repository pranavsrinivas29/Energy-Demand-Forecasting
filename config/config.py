import os
from skopt.space import Real, Integer

# --- Root Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Data Paths ---
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'smart_grid_dataset.csv')
FEATURE_ENGINEERED_DATA_PATH = os.path.join(DATA_DIR, 'feature_engineered.csv')
#PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed', 'clean_data.csv')
INFERENCE_DATA_PATH = os.path.join(DATA_DIR, 'inferenceset.csv')

# --- Model Paths ---
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'rf_model.pkl')

# --- Logging Path ---
LOG_DIR = os.path.join(BASE_DIR, 'logs')
LOG_PATH = os.path.join(LOG_DIR, 'training.log')

TRAIN_TEST_SPLIT = 0.2
# --- Hyperparameters (optional centralization) ---
RF_HYP_PARAMS = {
    'n_estimators': Integer(250, 600),
    'max_depth': Integer(3, 8),
    'min_samples_split': Integer(4, 12),
    'min_samples_leaf': Integer(12, 16),
    'max_features': Real(0.2, 0.6),
    'bootstrap': [True],
}

XGB_HYP_PARAMS = {
    'n_estimators': Integer(300, 700),          # more boosting rounds
    'max_depth': Integer(2, 8),                 # deeper trees
    'learning_rate': Real(0.001, 0.01, prior='log-uniform'),  # bigger steps
    'subsample': Real(0.1, 0.7),                 # less stochasticity
    'colsample_bytree': Real(0.4, 1.0),          # use more features per tree
    'min_child_weight': Integer(3, 7),           # allow finer splits
    'gamma': Real(0.0, 1.0),                     # weaker split penalty
    'reg_lambda': Real(0.3, 2.0, prior='log-uniform'),  # reduce L2
    'reg_alpha': Real(0.3, 2.0, prior='log-uniform'),   # reduce L1
}


META_LEARNING_HYP_PARAMS = {
    'hidden_layer_sizes': [(32, 16), (64, 32), (128, 64, 32)],
    'alpha': [1e-5, 1e-4, 1e-3],
    'learning_rate_init': [0.0005, 0.001, 0.005, 0.002, 0.0002, 0.0001],
    'activation': ['relu', 'tanh'],
}

GBR_HYP_PARAMS = {
    'n_estimators': Integer(400, 800),
    'max_depth': Integer(2, 7),
    'learning_rate': Real(0.0001, 0.1, prior='log-uniform'),
    'min_samples_split': Integer(4, 12),
    'min_samples_leaf': Integer(4, 8),
    'subsample': Real(0.5, 1.0)
}