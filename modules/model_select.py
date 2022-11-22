from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer

def rmse(y1,y2):
    return np.sqrt(np.mean(np.square(y2- y1)))

def corr(y1,y2):
    return pearsonr(y1,y2)[0]

def model_pipeline():
    pipelines = []
    pipelines.append(('Linear', Pipeline([('Scaler', StandardScaler()),('LR',LinearRegression())])))
    pipelines.append(('Lasso', Pipeline([('Scaler', StandardScaler()),('Lasso',Lasso())])))
    pipelines.append(('Ridge', Pipeline([('Scaler', StandardScaler()),('Ridge',Ridge())])))
    pipelines.append(('KNeighbor', Pipeline([('Scaler', StandardScaler()),('KNR',KNeighborsRegressor())])))
    pipelines.append(('GBR',Pipeline([('Scaler', StandardScaler()),('GBR',GradientBoostingRegressor())])))
    pipelines.append(('RF',Pipeline([('Scaler', StandardScaler()),('RF',RandomForestRegressor())])))
    pipelines.append(('XGB',Pipeline([('Scaler', StandardScaler()),('XGBoost',XGBRegressor())])))
    pipelines.append(('SVR-lin',Pipeline([('Scaler', StandardScaler()),('SVR-Linear',SVR(kernel = 'linear'))])))
    pipelines.append(('SVR-poly',Pipeline([('Scaler', StandardScaler()),('SVR-poly',SVR(kernel = 'poly'))])))
    pipelines.append(('SVR-rbf',Pipeline([('Scaler', StandardScaler()),('SVR-rbf',SVR(kernel = 'rbf'))])))
    pipelines.append(('MLP', Pipeline([('Scaler', StandardScaler()),('MLP',MLPRegressor())])))
    pipelines.append(('MLP2', Pipeline([('Scaler', StandardScaler()),('MLP',MLPRegressor((42,42,42),random_state=0))])))
    return pipelines

def get_scoring_dict():
    scoring_rmse = make_scorer(rmse, greater_is_better=False)
    scoring_r = make_scorer(corr, greater_is_better=True)
    scoring_dict = {'rmse':scoring_rmse,'r':scoring_r}
    return scoring_dict



