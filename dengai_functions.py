import numpy as np
import pandas as pd
from sklearn import preprocessing

''' ~~~~~~~~~~~~~~~~~~~~~~~ PREPROCESSING FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~ '''

def get_month(dates):
    ''' extracts month from the datasets'''
    dates = dates.str.split('-')
    month = []
    for date in dates:
        month.append(int(date[1]))
    month = pd.Series(month)
    return month

def standardise(x, scaler):
    ''' returns data scaled with scikit scaler as DataFrame (not numpy array)'''
    x_scaled = scaler.transform(x)
    x_scaled = pd.DataFrame(x_scaled, index=x.index, columns=x.columns)
    return x_scaled

def add_lagged_features(x, x_test, lagged_weeks):
    ''' for each column, appends a shifted column up to num_lags e.g. for
    var1, num_lags=2, appends var1(t-1), var1(t-2)
    '''

    # join train and test set
    x_temp = x.copy().append(x_test.copy(), ignore_index=True)

    # 
    x_new = x_temp.copy()
    for t in range(1, lagged_weeks+1):
        x_shifted = x_temp.shift(t)
        x_shifted.columns = x_temp.columns + '(t-' + str(t) + ')'
        for column_name in x_shifted:
            if 'cases' in column_name:
                x_shifted.rename(columns = {column_name:'cases(t-'+str(t+1)+')'}, inplace=True)
        x_new = x_new.join(x_shifted)
    
    # remove first values of train set that have nans
    x_new = x_new.dropna()
    x_new.reset_index(drop=True, inplace=True)

    train_length = len(x)-lagged_weeks
    x = x_new.iloc[0:train_length,:]
    x_test = x_new.iloc[train_length:len(x_new),:]
    x_test.reset_index(drop=True, inplace=True)

    return x, x_test
    

def preprocess(x, x_test, y, y_test=None, test=False, lagged_variables=False, lagged_weeks=0):
    ''' preprocess the data - sort columns, fill missing data, add lagged
    variable columns and standardise'''
    
    # reset the index of all datasets
    x = x.reset_index() 
    x_test = x_test.reset_index() 
    y = y.reset_index()
    
    # add month as feature, more useful than particular week
    x['month'] =  get_month(x['week_start_date'].copy())
    x_test['month'] = get_month(x_test['week_start_date'].copy())

    # drop unneeded columns
    x = x.drop(['city', 'weekofyear', 'week_start_date'], axis=1)
    x_test = x_test.drop(['city', 'weekofyear', 'week_start_date'], axis=1)
    y = y.drop(['city', 'weekofyear', 'year'], axis=1)
    
    # interpolate missing data, method='time' not needed as rows of data are 
    # equally spaced
    x = x.interpolate(method='linear')
    x_test = x_test.interpolate(method='linear')
    
    # add lagged columns
    # if lagged_weeks=0, only the dv is lagged (by a week)
    if lagged_variables: # for non time-series methods
        # add previous cases as a feature
        x['cases(t-1)'] = y.shift(1)
        x.fillna(method='bfill', inplace=True)
    
        x_test['cases(t-1)'] = 0
        x_test.loc[0, 'cases(t-1)'] = y.loc[y.index[-1], 'total_cases'].copy()
    
        x, x_test = add_lagged_features(x, x_test, lagged_weeks)
        y = y.iloc[lagged_weeks:, :]
        y.reset_index(drop=True,inplace=True)
        
    # standardise
    scaler = preprocessing.StandardScaler()
    scaler.fit(x)
    x = standardise(x, scaler)
    x_test = standardise(x_test, scaler)
       
    if test: # actual test data has no y_test
        return x, x_test, y, scaler
    else:
        y_test = y_test.reset_index()
        y_test = y_test.drop(['city', 'weekofyear', 'year'], axis=1)
        return x, x_test, y, y_test, scaler


''' ~~~~~~~~~~~~~~~~~~~~~~~ PREDICTION FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~ '''

def input_predictions(test_row, predictions, scaler):
    ''' 
    with columns such as cases(t-1), cases(t-2) etc. , for the recursive forecast, 
    predictions need to be input into the training set. predictions are 
    standardised with training statistics first then put in required position
    '''
    num_lags = int(list(test_row.columns.values)[-1].split('-')[1].split(')')[0])-1
    t = test_row.index[0]
    # get index of each 'cases' column (t-1), (t-2) etc...
    idxs = []
    for i in range(1,num_lags+2):
        idxs.append(test_row.columns.get_loc('cases(t-{})'.format(i)))
    # training data statistics for standardisation
    alpha = scaler.mean_
    sigma = np.sqrt(scaler.var_)
    # standardise and insert predictions into the test row
    if t < num_lags+1: # at this point we have actual value not prediction
        for i in range(1,t+1):
            test_row.loc[t, 'cases(t-{})'.format(i)] = (predictions[-i]-alpha[idxs[i-1]])/sigma[idxs[i-1]]
    else:
        for i in range(1,num_lags+2):
            test_row.loc[t, 'cases(t-{})'.format(i)] = (predictions[-i]-alpha[idxs[i-1]])/sigma[idxs[i-1]]       
    return test_row

def postprocess(predictions):
    ''' round predictions and convert to integer type'''
    predictions = np.array(predictions)
    predictions[predictions<0] = 0 # cases cannot be negative
    predictions = (np.round(predictions, 0)).astype(int)
    return predictions

def recursive_forecast(x, x_test, y, model, scaler, is_mlp):
    ''' for multistep forecasting
    makes a prediction on test row, then appends test row and prediction
    to training sets'''
    predictions = []
    for t in range(len(x_test)):
        test_row = x_test.loc[[t]].copy() # row on which prediction is made
        if t > 0: # after first prediction, need to use prediction and update the model    
            test_row = input_predictions(test_row, predictions, scaler) # input the previous predictions
            if is_mlp: # update weights if mlp
                model.partial_fit(x, y)
            else: # retrain if other model
                model.fit(x, y)
        yhat = model.predict(test_row)[0]
        predictions.append(yhat)
        x = x.append(test_row, ignore_index=True)
        y = np.append(y, yhat)
    return predictions
    
def recursive_forecast_keras(x, x_test, y, model, scaler):
    ''' same as above but data needs reshaping into 3d'''
    x = x.values.reshape((x.shape[0], 1, x.shape[1]))
    y = y.values
    model.fit(x, y, epochs=200, batch_size=1, verbose=0)
    predictions = []
    for t in range(len(x_test)): # go through each row of val data
        test_row = x_test.loc[[t]].copy()
        if t > 0: # append previous prediction as cases_prev_wk for current week
            test_row = input_predictions(test_row, predictions, scaler) 
            model.train_on_batch(x,y)
        test_row = test_row.values.reshape((test_row.shape[0], 1, test_row.shape[1]))
        yhat = model.predict(test_row)[0][0]
        predictions.append(yhat)        
        x = np.append(x, test_row, axis=0) 
        y = np.append(y, yhat)
    return predictions 