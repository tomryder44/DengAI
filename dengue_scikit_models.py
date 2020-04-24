import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

# custom function imports
from dengai_functions import preprocess
from dengai_functions import recursive_forecast
from dengai_functions import postprocess

# load datasets
x_train = pd.read_csv('dengue_features_train.csv', index_col=[0])
y = pd.read_csv('dengue_labels_train.csv', index_col=[0])

# create dataset for San Juan
sj  = x_train.loc['sj']
sj_y = y.loc['sj']

# only keep more recent data
#sj = sj.iloc[(len(sj)-(52*5)):,:]
#sj_y = sj_y.iloc[(len(sj_y)-(52*5)):,:] 

# split into train and validation sets
sj, sj_val, sj_y, sj_y_val = train_test_split(sj, sj_y, test_size=0.5, 
                                                shuffle=False)

# repeat for iquitos
iq  = x_train.loc['iq']
iq_y = y.loc['iq']
iq, iq_val, iq_y, iq_y_val = train_test_split(iq, iq_y, test_size=0.5, 
                                                shuffle=False)

def select_model(x, x_test, y, y_test, models, grids, scaler):
    ''' for each model, carry out grid search with recursive_forecast function
    selects the model with the lowest MAE on the validation set
    '''
    best_score = 1000 # lowest MAE on all models for all types of model
    for model, grid in zip(models, grids): 
        named_param = list(grid.keys())[0] # parameter of grid search
        is_mlp = isinstance(model, MLPRegressor)
        best_model_score = 1000 # lowest MAE for all models for each type of model
        optimal_value_found = False
        g = grid[named_param]
        while optimal_value_found is False:
            scores = []
            for value in g:
                param = {named_param:value}
                model.set_params(**param) # set hyperparameter value
                model.fit(x, y)
                predictions = recursive_forecast(x, x_test, y, model, scaler, is_mlp)
                predictions = postprocess(predictions)
                score = mean_absolute_error(y_test, predictions)
                scores.append(score)
                
                if score < best_model_score:
                    best_model_predictions = predictions
                    best_model_score = score
                
                if score < best_score:
                    best_score = score
                    best_predictions = predictions
                    best_model = model.get_params()
                
                if value == g[-1]:  
                    # plot results of grid search, checks that range is correct
                    plt.plot(g, scores)
                    plt.title('Grid search results for %s' % model.__class__.__name__)
                    plt.xlabel(named_param)
                    plt.ylabel('MAE')
                    plt.show()
                    # if best score is from value at end, define anew grid search
                    # range and carry out new search
                    if min(scores) == scores[-1]:
                        print('Optimal value not found. Restarting grid search with new range')
                        g = np.linspace(g[-1], (g[-1])*2, len(g)).astype(int)
                    else:
                        optimal_value_found = True
                
        # for each type of model, plot the best set of predictions
        plt.plot(y_test, label='actual')
        plt.plot(best_model_predictions, label='predictions')
        plt.xlabel('time')
        plt.ylabel('cases')
        plt.legend()
        plt.title('Lag=%d, Model=%s, MAE=%.2f' % (int(list(x.columns.values)[-1].split('-')[1].split(')')[0])-1, model.__class__.__name__, best_model_score))
        plt.show()
    return best_model, best_predictions, best_score

def optimise_all(x, x_test, y, y_test, models, grids):
    ''' use select_model trying different number of lagged weeks to find best 
    overall predictions'''
    lags_list = [1, 2, 3, 4]
    best_score = 100
    for lag in lags_list:
        x_lagged, x_test_lagged, y_lagged, y_test_lagged, scaler = preprocess(
            x, x_test, y, y_test, lagged_variables=True, lagged_weeks=lag)
        model, predictions, score = select_model(x_lagged, x_test_lagged, 
            y_lagged.values.ravel(), y_test_lagged, models, grids, scaler)
        if score < best_score:
            best_score = score
            best_model = model
            best_predictions = predictions
            best_lag = lag
    return best_model, best_lag, best_predictions

# ~~ Ridge ~~
ridge_grid = {'alpha':np.linspace(10**-2, 10**1, 20)}
ridge = Ridge()
    
# ~~ KNN ~~
knn_grid = {'n_neighbors':np.linspace(1, 20, 20).astype(int)}
knn = KNeighborsRegressor()

# ~~ XGBoost ~~
xgb_grid = {'n_estimators':np.linspace(1, 10, 10).astype(int)}
xgb = XGBRegressor(objective='reg:squarederror')

# ~~ MLP ~~
mlp_grid = {'alpha':np.linspace(10**-2, 10**1, 15)}
mlp = MLPRegressor(hidden_layer_sizes=(200,), batch_size=1)
        
models = [ridge, knn, xgb, mlp]
grids = [ridge_grid, knn_grid, xgb_grid, mlp_grid]

sj_model, sj_lag, sj_predictions = optimise_all(sj, sj_val, sj_y, sj_y_val, models, grids)
iq_model, iq_lag, iq_predictions = optimise_all(iq, iq_val, iq_y, iq_y_val, models, grids)
