import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config import FEATURE_ENGINEERED_DATA_PATH, TRAIN_TEST_SPLIT, RF_HYP_PARAMS, XGB_HYP_PARAMS, META_LEARNING_HYP_PARAMS, GBR_HYP_PARAMS
from typing import Tuple

from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
from sklearn.ensemble import GradientBoostingRegressor


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
# Configure logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class EnergyDemandForecasting:
    def __init__(self, target='load_da'):
        self.train_test_split = TRAIN_TEST_SPLIT
        self._target = target
        self._model_define()
        self.tscv = TimeSeriesSplit(n_splits=6)
        self.quantiles = [0.1, 0.5, 0.9]
        
    @property
    def target(self):
        return self._target
    
    @target.setter
    def target(self, value):
        self._target=value
        
    def _model_define(self):
        self.rf = RandomForestRegressor(random_state=42, n_jobs=-1, oob_score=True)
        self.xgb = XGBRegressor(objective='reg:squarederror',random_state=42,n_jobs=-1,
            tree_method='hist', eval_metric='rmse')
        self.meta_MLP = MLPRegressor(solver='adam', max_iter=3000, random_state=42)
        self.gbr_base = GradientBoostingRegressor(loss="quantile", alpha=0.5, random_state=42)

    #raw data load
    def data_loading(self, FEATURE_ENGINEERED_DATA_PATH:str)->pd.DataFrame:
        df = pd.read_csv(FEATURE_ENGINEERED_DATA_PATH)
        df.set_index('Timestamp', inplace=True)
        return df
    
    #feature engineering
    def preprocessing(self, df: pd.DataFrame)->pd.DataFrame:
        df['load_da'] = df['Predicted Load (kW)'].shift(-1)
        df['load_d2'] = df['Predicted Load (kW)'].shift(-2)
        
        df['lag_1'] = df['Predicted Load (kW)'].shift(1)
        df['lag_2'] = df['Predicted Load (kW)'].shift(2)
        df['lag_3'] = df['Predicted Load (kW)'].shift(3)
        
        df['pct_tmp_chg'] = df['Temperature (°C)'].pct_change()
        df['pow_cons_chg'] = df['Power Consumption (kW)'].pct_change()
        df['roll_mean_load3'] = df['Predicted Load (kW)'].rolling(3).mean()
        df['roll_std_load3'] = df['Predicted Load (kW)'].rolling(3).std()
        
        df.to_csv('inference_featured.csv')
        return df
    
    def featured_data_loading(self, FEATURE_ENGINEERED_DATA_PATH:str)->pd.DataFrame:
        df = pd.read_csv(FEATURE_ENGINEERED_DATA_PATH)
        df.set_index('Timestamp', inplace=True)
        return df
    
    def get_model_split_training_datasets(self, df:pd.DataFrame)-> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        train_size = int((1 - self.train_test_split) * len(df))
        df_train = df.iloc[:train_size]
        df_test  = df.iloc[train_size:]
        
        #tune_split = int(0.8 * len(df_train))
        #train_tune = df_train.iloc[:tune_split]
        #val_tune   = df_train.iloc[tune_split:]
        
        X_train_tune = df_train.drop(self.target, axis=1)
        y_train_tune = df_train[self.target]

        #X_val_tune = val_tune.drop(self.target, axis=1)
        #y_val_tune = val_tune[self.target]
        
        X_test_tune = df_test.drop(self.target, axis=1)
        y_test_tune = df_test[self.target]
        
        return X_train_tune, X_test_tune, y_train_tune, y_test_tune

    def get_model_full_training_datasets(self, df:pd.DataFrame)-> Tuple[pd.DataFrame, pd.Series]:
        X = df.drop(self.target, axis=1)
        y = df[self.target]
        return X, y
    
    #for model training tune get best hyperparams
    def model_training(self, **kwargs):
        X_train = kwargs.get('X_train')
        X_val = kwargs.get('X_val')
        X_test = kwargs.get('X_test')
        y_train = kwargs.get('y_train')
        y_val = kwargs.get('y_val')
        y_test = kwargs.get('y_test')
        model_name = kwargs.get('model_name')

        if model_name is None:
            raise Exception('Model name not found')
        elif model_name == 'rf':
            logger.info('Random Forest Model Training')
            param_space = RF_HYP_PARAMS

            opt = BayesSearchCV(
                estimator=self.rf,
                search_spaces=param_space,
                n_iter=30,
                cv=self.tscv,
                scoring='neg_root_mean_squared_error',
                n_jobs=-1,
                random_state=42,
                verbose=0
            )

            opt.fit(X_train, y_train)
            #print("Best Parameters:", opt.best_params_)
            #print("Best CV Score:", -opt.best_score_)
            logger.info('Random Forest Model Trained')

            return opt
        
        elif model_name == 'xgb':
            param_space = XGB_HYP_PARAMS

            opt = BayesSearchCV(
                estimator=self.xgb, search_spaces=param_space,
                n_iter=50, cv=self.tscv,
                scoring='neg_root_mean_squared_error', n_jobs=-1,
                random_state=42, verbose=1)
            
            opt.fit(X_train, y_train)
            return opt
        
        elif model_name == 'meta':
            meta_opt_mlp = GridSearchCV(estimator=self.meta_MLP,
            param_grid= META_LEARNING_HYP_PARAMS,cv=self.tscv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,verbose=1)

            meta_opt_mlp.fit(X_train, y_train)
            return meta_opt_mlp
        
        elif model_name == 'qrf':
            pass
        else:
            gbr_opt = BayesSearchCV(
            estimator=self.gbr_base,
            search_spaces=param_space,
            n_iter=40,
            cv=self.tscv,
            scoring='neg_root_mean_squared_error',
            n_jobs=1,
            random_state=42,
            verbose=2)

            gbr_opt.fit(X_train, y_train)

            #print("Best Median Model Parameters:", gbr_opt.best_params_)
            #print("Best Median CV RMSE:", -gbr_opt.best_score_)
            return gbr_opt
        
    def model_predict(self, Xtest:pd.DataFrame, model)->pd.Series:
        ypred = model.predict(Xtest)
        return ypred
    
    #eval metrics during split training
    def model_eval(self, ytest: pd.Series, ypred: pd.Series)->Tuple[float, float]:
        mae = mean_absolute_error(ytest, ypred)
        rmse = mean_squared_error(ytest, ypred, squared=False)
        return mae, rmse
        
if __name__=='__main__':
    dp = EnergyDemandForecasting()
    data = dp.featured_data_loading(FEATURE_ENGINEERED_DATA_PATH=FEATURE_ENGINEERED_DATA_PATH)
    #print(data.head(3))
    X, y = dp.get_model_full_training_datasets(data)
    mod = dp.model_fulltraining(X, y, 'rf')
    print(mod.best_params_)
    