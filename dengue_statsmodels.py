# statsmodels
import statsmodels.api as sm
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from dengai_functions import preprocess
from dengai_functions import postprocess

# load datasets
x_train = pd.read_csv('dengue_features_train.csv', index_col=[0])
x_test = pd.read_csv('dengue_features_test.csv', index_col=[0])
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

sj, sj_val, sj_y, sj_y_val, sj_scaler = preprocess(sj, sj_val, sj_y, sj_y_val)
iq, iq_val, iq_y, iq_y_val, iq_scaler = preprocess(iq, iq_val, iq_y, iq_y_val)

def arimax(x, x_test, y, y_test):
    ''' tries each combo of parameters for arimax model and returns the best
    '''
    min_mae = 100
    best_predictions = []
    for p in range(1,10):
        for d in range(2,3):
            for q in range(0,10):
                model = sm.tsa.statespace.SARIMAX(endog=y, exog=x, order=(p,d,q),
                                                  time_varying_regression=True, 
                                                  mle_regression=False)
                model_fit = model.fit()
                predictions = model_fit.predict(start=len(x), end=len(x)+len(x_test)-1, exog=x_test)
                predictions = postprocess(predictions)
                mae = mean_absolute_error(y_test, predictions)
                if mae < min_mae:
                    min_mae = mae
                    best_params = [p,d,q]
                    best_predictions = predictions
                # plot predictions for each model    
                plt.plot(y_test, label='actual') 
                plt.plot(predictions, label='predictions')
                plt.title('(%d,%d,%d), MAE=%.2f'%(p,d,q,mae))
                plt.legend()
                plt.xlabel('time')
                plt.ylabel('cases')
                plt.show()
    return best_params, best_predictions

sj_params, sj_predictions = arimax(sj, sj_val, sj_y, sj_y_val)
iq_params, iq_predictions = arimax(iq, iq_val, iq_y, iq_y_val)
