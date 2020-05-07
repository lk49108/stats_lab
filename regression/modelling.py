'''
import os as os
os.chdir('stats_lab')
sys.path.insert(0, os.getcwd())
'''

import data_scheduler.lib_data_merger as mice_data
import matplotlib.pyplot as plt
import os as os
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.multivariate.manova import MANOVA



#First we want to fetch the relevant data which includes the brain and running activity
mice_data_dir = r'C:\Users\Massimo\Documents\StatsLab\New Data'
md = mice_data.MiceDataMerger(mice_data_dir)
treatments = ['glu', 'eth', 'sal', 'nea']
signals = ['brain_signal', 'running', 'v_o2', 'v_co2', 'rq', 'heat']

#Envelope function
def envelope(df, window_size=500):

    # We need to smoothen the time series for easier feature extraction
    # (the larger the window the smoother it will be)
    windows = df.rolling(window=window_size)
    df_smooth = windows.mean()

    # plt.plot(brain_smooth.iloc[:,1], )
    # plt.plot(brain.iloc[:,1])
    # plt.show()

    # Rectification (all time points are positive)
    df_rectified = df_smooth.apply(np.abs)
    df_envelope = df_rectified.rolling(50).mean()
    df_envelope = df_envelope.dropna()
    return df_envelope

#Ridge regression with polynomial features d=3
def get_ridge_coef(brain, running, plot_prediction = False, include_brain_signal = False):
    # We get the envelopes
    brain_envelope = envelope(brain)
    running_envelope = envelope(running)

    X = pd.concat([brain_envelope.iloc[:, 1], brain_envelope.iloc[:, 1] ** 2,
                   brain_envelope.iloc[:, 1] ** 3], axis=1)
    y = running_envelope

    # Before we fit the model, we standardize the values within the range [0,1]
    mm = MinMaxScaler()

    mm.fit(X)
    X = mm.transform(X)
    X = pd.DataFrame(X).set_index(running_envelope.index)
    mm = MinMaxScaler()

    mm.fit(y)
    y = mm.transform(y)
    y = pd.DataFrame(y).set_index(running_envelope.index)

    # We fit the model with the Ridge linear regression model
    ridge = Ridge(normalize=True, fit_intercept=False)
    ridge.fit(X, y)
    prediction = pd.DataFrame(ridge.predict(X)).set_index(running_envelope.index)

    if plot_prediction:
        # Transform brain signal to plot it
        mm = MinMaxScaler()
        mm.fit(np.array(brain_envelope.iloc[:, 1]).reshape(-1,1))
        brain_envelope_transformed = mm.transform(np.array(brain_envelope.iloc[:, 1]).reshape(-1,1))
        brain_envelope_transformed = pd.DataFrame(brain_envelope_transformed).set_index(running_envelope.index)

        # We plot the prediction of the running signal, the actual running signal and the regressor (brain signal)
        plt.figure()
        plt.plot(prediction, label='Prediction')
        if include_brain_signal:
            plt.plot(brain_envelope_transformed, label='Brain signal (envelope)')
        plt.plot(y, label='Running signal (envelope) of' + ' ' + str(mouse_ids[j]) + ' ' + treatments[i])
        plt.title('Time Series Regression: Treatment' + ' ' + treatments[i] + ' ' + 'of' + ' ' + str(mouse_ids[j]))
        plt.legend()
        plt.show()

        # We look at the residuals
        # residuals[i] = prediction - y
    return ridge.coef_

#Slicing of data according to upper and lower bound
def preprocessing(signal, min_lowerbound, min_upperbound):
    signal_sliced = signal.sliced_data(min_lowerbound).left_slice(min_upperbound).get_pandas()
    signal_sliced = signal_sliced.set_index('time_min')
    return signal_sliced


#Defining global variables
mouse_ids_list = [165, 166, 167, 126, 170, 299, 302, 303, 306, 307, 323, 327]
mouse_ids = mouse_ids_list[9:12] #(to select all mice use [0:12])
objects = ('glu', 'eth', 'sal', 'nea')
df_anova_coef = pd.DataFrame(columns = ['coef1','coef2','coef3','Treatment'])

#Defining parameters
option_acf = False #set to True if want to plot acf
option_pacf = False #set to True if want to plot acf
lags_correlogram = 100 #set lag size for correlogram

option_scatter = False #set to True if want to plot scatterplot of brain vs. running activity
option_plot_prediction = True #set to True if want to plot prediction vs. actual signal

#Interval in minutes
lower = 30
upper = 90


for j in range(len(mouse_ids)):
    for i in range(len(treatments)):
        brain_signal = md.fetch_mouse_signal(mouse_id = mouse_ids[j], treat=treatments[i], signal='brain_signal')
        running_signal = md.fetch_mouse_signal(mouse_id = mouse_ids[j], treat=treatments[i], signal='running')

        if brain_signal is not None and running_signal is not None:
            if type(brain_signal) is not list and type(running_signal) is not list:

                brain = preprocessing(brain_signal, lower, upper)
                running = preprocessing(running_signal, lower, upper)

                coefficients = np.abs(get_ridge_coef(brain, running, option_plot_prediction))

                df_anova_coef = df_anova_coef.append(pd.DataFrame({ 'coef1': coefficients[0:,0],
                                                                    'coef2': coefficients[0:,1],
                                                                    'coef3': coefficients[0:,2],
                                                                    'Treatment': [treatments[i]]}))
                print(coefficients)

            elif type(brain_signal) is list and type(running_signal) is list:
                for m in range(len(brain_signal)):
                    brain = preprocessing(brain_signal[m], lower, upper)
                    running = preprocessing(running_signal[m], lower, upper)

                    coefficients = np.abs(get_ridge_coef(brain, running, option_plot_prediction))
                    df_anova_coef = df_anova_coef.append(pd.DataFrame({'coef1': coefficients[0:,0],
                                                                       'coef2': coefficients[0:,1],
                                                                       'coef3': coefficients[0:,2],
                                                                       'Treatment': [treatments[i]]}))
        else:
            print(mouse_ids[j], treatments[i])



#We perform a simple one-way ANOVA for optimal R values
maov = MANOVA.from_formula('coef1 + coef2 + coef3  ~ Treatment', data=df_anova_coef)
print(maov.mv_test())

