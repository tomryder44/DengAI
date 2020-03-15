import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# keras for LSTM
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.regularizers import l2

from dengai_functions import preprocess
from dengai_functions import recursive_forecast_keras
from dengai_functions import postprocess

# load datasets
x_train = pd.read_csv('dengue_features_train.csv', index_col=[0])
y = pd.read_csv('dengue_labels_train.csv', index_col=[0])

# create dataset for San Juan
sj  = x_train.loc['sj']
sj_y = y.loc['sj']
sj, sj_val, sj_y, sj_y_val = train_test_split(sj, sj_y, test_size=0.5, 
                                                shuffle=False)

# create dataset for Iquitos
iq  = x_train.loc['iq']
iq_y = y.loc['iq']
iq, iq_val, iq_y, iq_y_val = train_test_split(iq, iq_y, test_size=0.5, 
                                                shuffle=False)

def lstm_gridsearch(x, x_test, y, y_test, scaler):
    '''gridsearch for optimal L2 regularisation value with recursive_forecast
    selects L2 with lowest MAE'''
    best_score = 100
    alpha_list = np.linspace(10**-2, 10**1, 10)
    scores = []
    optimal_value_found = False
    poo = alpha_list
    while optimal_value_found is False:
            scores = []
            for alpha in poo:
                model = Sequential()
                model.add(LSTM(200, input_shape=(1, x.shape[1]),
                    kernel_regularizer=l2(alpha), 
                    recurrent_regularizer=l2(alpha), 
                    bias_regularizer=l2(alpha)))
                model.add(Dense(1, activation=None))
                ad = optimizers.Adam(learning_rate=0.0001)
                model.compile(loss='mean_absolute_error', optimizer=ad)
                predictions = recursive_forecast_keras(x, x_test, y, model, scaler)
                predictions = postprocess(predictions)
                score = mean_absolute_error(y_test, predictions) 
                scores.append(score)
                # plot predictions of each alpha
                plt.plot(y_test, label='actual')
                plt.plot(predictions, label='predictions')
                plt.title('Lag=%d, Alpha=%f, MAE=%.2f' % (int(list(x.columns.values)[-1].split('-')[1].split(')')[0])-1, alpha, score))
                plt.xlabel('time')
                plt.ylabel('cases')
                plt.legend()
                plt.show()
                if score < best_score:
                    best_score = score
                    best_alpha = alpha
                    best_predictions = predictions
                if alpha == poo[-1]:  
                    # plot results of grid search, checks that range is correct
                    plt.plot(poo, scores)
                    plt.title('Grid search results for LSTM')
                    plt.xlabel('Alpha')
                    plt.ylabel('MAE')
                    plt.show()
                    # if best score is from value at end, define a new grid search
                    # range and carry out new search
                    if min(scores) == scores[-1]:
                        print('Optimal value not found. Restarting grid search with new range')
                        poo = np.linspace(poo[-1], (poo[-1])*2, len(poo))
                    else:
                        optimal_value_found = True
        
    return best_alpha, best_predictions, best_score

def optimise_all(x, x_test, y, y_test):
    ''' use lstm_gridsearch trying different number of lagged weeks to find best 
    overall predictions'''
    lags_list = [1, 2, 3, 4]
    best_score = 100
    best_lag = 0
    for lag in lags_list:
        x_lagged, x_test_lagged, y_lagged, y_test_lagged, scaler = preprocess(
            x, x_test, y, y_test, lagged_variables=True, lagged_weeks=lag)
        alpha, predictions, score = lstm_gridsearch(x_lagged, x_test_lagged, 
                                                y_lagged, y_test_lagged, scaler)
        if score < best_score:
            best_score = score
            best_alpha = alpha
            best_predictions = predictions
            best_lag = lag
    return best_alpha, best_lag, best_predictions
        
sj_alpha, sj_lag, sj_predictions = optimise_all(sj, sj_val, sj_y, sj_y_val)
iq_alpha, iq_lag, iq_predictions = optimise_all(iq, iq_val, iq_y, iq_y_val)
