'''
import os as os
os.chdir('stats_lab')
sys.path.insert(0, os.getcwd())
'''

import data_scheduler.lib_data_merger as mice_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import pingouin as pg
from sklearn.preprocessing import binarize
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import RFE

from statsmodels.multivariate.manova import MANOVA
from sklearn.metrics import recall_score
#First we want to fetch the relevant data which includes the brain and running activity
mice_data_dir = r'C:\Users\Massimo\Documents\StatsLab\New Data'
md = mice_data.MiceDataMerger(mice_data_dir)
treatments = ['glu', 'eth', 'sal', 'nea']
signals = ['brain_signal', 'running', 'v_o2', 'v_co2', 'rq', 'heat']

#We want to shift the brain signal into the set of real positive numbers to make
#the times series regression more coherent
def shift(df):
    df_shifted = df + 3
    return df_shifted

#Logistic regression model
def get_relevant_features(brain, running):

    #We remove first the NaNs created though the shift of the time series
    X = brain.iloc[feature_shift:,]
    y = running[feature_shift:,]

    # We fit a logistic regression model with the SGDClassifier
    estimator = SGDClassifier(loss='log', penalty='l1', alpha=0.0001, l1_ratio=0.15,
                        fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True,
                        verbose=0, epsilon=0.1, n_jobs=None, random_state=None,
                        learning_rate='optimal', eta0=0.0, power_t=0.5,
                        early_stopping=False, validation_fraction=0.1,
                        n_iter_no_change=5, class_weight='balanced', warm_start=False, average=False)
    selector = RFE(estimator, n_features_to_select=3, step=1)
    selector = selector.fit(X, y)

    return selector.support_

def plot_prediction(prediction, brain, running, include_brain_signal = False):
    X = brain.iloc[:,hemisphere]
    y = running

    # We plot the prediction of the running signal, the actual running signal and the regressor (brain signal)
    plt.figure()
    plt.plot(prediction, label='Prediction')
    if include_brain_signal:
        plt.plot(X, label='Brain signal (shift)')
    plt.plot(y, label='Running signal (shift) of' + ' ' + str(mouse_ids[j]) + ' ' + treatments[i])
    plt.title('Time Series Regression: Treatment' + ' ' + treatments[i] + ' ' + 'of' + ' ' + str(mouse_ids[j]))
    plt.xlabel(xlabel='Minutes')
    plt.ylabel(ylabel='Signal value')
    plt.legend()
    plt.show()

#Slicing of data according to upper and lower bound
def preprocessing(signal, min_lowerbound, min_upperbound):
    signal_sliced = signal.sliced_data(min_lowerbound).left_slice(min_upperbound).get_pandas()
    signal_sliced = signal_sliced.set_index('time_min')
    return signal_sliced

def create_features(brain, feature_shift):
    brain = brain.iloc[:,[hemisphere]]

    for i in range(1,feature_shift):
        brain.insert(loc=i, column= str(i), value = brain.iloc[:,0].shift(i).values)

    return brain

#Defining global variables
mouse_ids_list = [165, 166, 167, 168, 126, 170, 299, 302, 303, 306, 307, 323, 327]
mouse_ids = mouse_ids_list[0:1] #(to select all mice use [0:13])


option_scatter = False #set to True if want to plot scatterplot of brain vs. running activity
option_plot_prediction = False #set to True if want to plot prediction vs. actual signal
option_brain_signal = False #set to True if want to plot brain signal as well

#Interval in minutes
lower = 30
upper = 90

#Parameters
feature_shift = 20 #Features time shift of brain signal (i.e. 10 = 10 features with 0 up to shift of 9)
running_cutoff = 10.0 #When is the mouse considered to be in the running state (e.g. 10cm/s)
hemisphere = 1 #put 0 for left brain hemisphere, put 1 for right brain hemisphere

#We create a df for collecting all the results
cols = [str(x) for x in range(0,feature_shift)]
lst_values = []
lst_treatments = []

for j in range(len(mouse_ids)):
    for i in range(len(treatments)):
        brain_signal = md.fetch_mouse_signal(mouse_id = mouse_ids[j], treat=treatments[i], signal='brain_signal')
        running_signal = md.fetch_mouse_signal(mouse_id = mouse_ids[j], treat=treatments[i], signal='running')


        if brain_signal is not None and running_signal is not None:
            if type(brain_signal) is not list and type(running_signal) is not list:

                brain = preprocessing(brain_signal, lower, upper)
                brain = shift(brain)
                brain = create_features(brain, feature_shift)
                running = preprocessing(running_signal, lower, upper)
                running_bin = binarize(X=running, threshold= running_cutoff, copy=True)

                feature_array = get_relevant_features(brain, running_bin)
                lst_values.append(feature_array)
                lst_treatments.append(treatments[i])


                #if option_plot_prediction:
                    #plot_prediction(prediction=prediction, brain=brain, running=running_bin, include_brain_signal=False)

            elif type(brain_signal) is list and type(running_signal) is list:
                for m in range(len(brain_signal)):
                    brain = preprocessing(brain_signal[m], lower, upper)
                    brain = shift(brain)
                    brain = create_features(brain, feature_shift)

                    running = preprocessing(running_signal[m], lower, upper)
                    running_bin = binarize(X=running, threshold= running_cutoff, copy=True)

                    feature_array = get_relevant_features(brain, running_bin)
                    lst_values.append(feature_array)
                    lst_treatments.append(treatments[i])

        else:
            print(mouse_ids[j], treatments[i])


df1 = pd.DataFrame(lst_values, columns=cols)
df2 = pd.DataFrame(lst_treatments, columns=['Treatments'])
summary = pd.concat([df1, df2], axis=1)
summary_grouped = summary.groupby('Treatments').sum()

for i in (summary_grouped.index):
    summary_grouped_per = summary_grouped.loc[i] / (summary_grouped.loc[i].sum())
    plt.figure()
    plt.bar(summary_grouped.columns, summary_grouped_per, align='center', alpha=0.5)
    #plt.xticks(y_pos, objects)
    plt.ylabel('Relative Frequency')
    plt.title('Distribution of selected feature with different lags for treatment: ' + i)
    plt.show()