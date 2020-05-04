'''
import os as os
os.chdir('stats_lab')
sys.path.insert(0, os.getcwd())
'''

import data_scheduler.lib_data_merger as mice_data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#First we want to fetch the relevant data which includes the brain and running activity
mice_data_dir = r'C:\Users\Massimo\Documents\StatsLab'
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

# We import all the files from the folder
name_list = os.listdir(r"C:\Users\Massimo\Documents\StatsLab\New Data")
print(type(name_list))
print(name_list)

new_list = []

for i in name_list:
    new_list.append(i[0:3])

new_set = set(new_list)

model = [0,0,0,0]
equal, unequal = np.zeros(4), np.zeros(4)
residuals = [[0],[0],[0],[0]]
mouse_ids = [165,166,126, 168, 170, 299, 302, 303, 306, 307, 323, 327]
objects = ('glu', 'eth', 'sal', 'nea')
lag_size = 1 #for % changes of time series comparison

for j in range(len(mouse_ids)):
    for i in range(len(treatments)):
        brain = md.fetch_mouse_signal(mouse_id = mouse_ids[j], treat=treatments[i], signal='brain_signal').sliced_data(30).get_pandas()
        running = md.fetch_mouse_signal(mouse_id = mouse_ids[j], treat=treatments[i], signal='running').sliced_data(30).get_pandas()
        heat = md.fetch_mouse_signal(mouse_id = mouse_ids[j], treat=treatments[i], signal='heat').sliced_data(30).get_pandas()

        brain = brain.set_index('time_min')
        running = running.set_index('time_min')
        heat = heat.set_index('time_min')

        #We get the envelopes
        brain_envelope = envelope(brain)
        running_envelope = envelope(running)
        heat_envelope = envelope(heat)

        print('I got mouse id'+str(mouse_ids[i]))

        '''

        # Now we fit a linear regression model and calculate it's cv score
        X = pd.concat([brain_envelope.iloc[:, 1], brain_envelope.iloc[:, 1]**2,
        brain_envelope.iloc[:, 1]**3], axis=1)
        y = running_envelope

        #Before we fit the model, we standardize the values within the range [0,1]

        mm = MinMaxScaler()
        mm.fit(X)
        X = mm.transform(X)
        X = pd.DataFrame(X).set_index(running_envelope.index)

        mm = MinMaxScaler()
        mm.fit(y)
        y = mm.transform(y)
        y = pd.DataFrame(y).set_index(running_envelope.index)

        #We fit the model with the Ridge linear regression model
        ridge = Ridge()
        model[i] = ridge.fit(X,y)
        prediction = pd.DataFrame(ridge.predict(X)).set_index(running_envelope.index)
        #We look at the scores
        scores = cross_val_score(Ridge(), X, y, cv=5)
        print(scores); print(treatments[i])

        #We look at the residuals
        residuals[i] = prediction - y

        #We plot the prediction vs. the actual values
        plt.figure()
        plt.plot(prediction, label = 'Prediction')
        plt.plot(y, label = 'Running Activity of'+' '+str(mouse_ids[j])+' '+treatments[i])
        #plt.plot(X.set_index(running_envelope.index), label = 'Brain Activity of 165'+' '+i)
        plt.title('Time Series Regression: Treatment'+' '+ treatments[i]+' '+'of'+' '+str(mouse_ids[j]))
        plt.legend()
        plt.show()


        #We output a plot to see if all changes in % of both time series have the same sign
        equal_sign = ((prediction.pct_change(periods=lag_size)) * (y.pct_change(periods=lag_size))).dropna()
        equal[i] = ((equal_sign >= 0).sum()/len(equal_sign))
        unequal[i] = ((equal_sign < 0).sum()/len(equal_sign))

        if i+1 == len(treatments):
            plt.figure()
            y_pos = np.arange(len(treatments))
            plt.bar(y_pos, equal, align= 'center')
            plt.xticks(y_pos, objects)
            plt.title('Equality of sign for % change of '+str(mouse_ids[j]))
            plt.ylim(top=1)

            #plt.figure()
            #plt.bar(y_pos, unequal, align= 'center')
            #plt.xticks(y_pos, objects)

            # We look at the distribution of the residuals
            plt.figure()
            for i in range(len(residuals)):

                plt.hist(residuals[i].iloc[:, 0], bins=500, label='Residuals of TS Regression' + ' ' + treatments[i])
                plt.legend()
                plt.title('Residuals of TS Regression of '+str(mouse_ids[j]))
                plt.show()

#Next we want to collect the regression coefficients for all mice and then contrast them against
#the saline treatment to see if they are significantly different
#For this we need to first load the full data of all mice, extract the coefficients and then compare them
#e.g. via a two-sample t-test




'''
#Now we implement a function that takes a series as an input and outputs the % change with a given lag level
def percentage(series, lag_level=1):
    perc = np.array([x for x in range(len(series))])
    for i in range(len(series)-lag_level):
        perc[i] = ((pd.Series.to_numpy(series)[i+lag_level]/pd.Series.to_numpy(series)[i])-1)
    return perc
    
#Look at the distribution of the raw data

for i in range(len(treatments)):
    brain = md.fetch_mouse_signal(mouse_id = 166, treat=treatments[i], signal='brain_signal').get_pandas()
    running = md.fetch_mouse_signal(mouse_id = 166, treat=treatments[i], signal='running').get_pandas()
    heat = md.fetch_mouse_signal(mouse_id = 166, treat=treatments[i], signal='heat').get_pandas()

    brain = brain.set_index('time_min')
    running = running.set_index('time_min')
    plt.figure()
    plt.hist(brain.iloc[:,1], bins = 100)
    plt.show()
'''

'''
All about ts regression:

Plotting time series data

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12,6))
data.plot('date','close',ax=ax)
ax.set(title='title')

Time series analysis for auditory data:
import librosa as lr

audio, sfreq = lr.load('file')

Classification of time series

Smoothing time series

window_size = 50 (the larger the window the smoother it will be)
windows = audio.rolling(window=window_size)
aduio_smooth = windowed.mean()

Rectification (all time points are positive)
audio-rectified = audio.apply(np.abs)
audio_envelope = audio_rectified.rolling(50).mean()

After we have the envelope of the time series we can do the
real feature engineering

envelope_mean = np.mean(audio_envelope, axis = 0)
...
envelope_max = np.max(aduio_envelope, axis = 0)

#Now we create our training data for a classifier
X = np.column_stack([envelope_mean, ..., envelope_max])

y = labels.reshape([-1,1])

Predicting data over time

Interpolation in Pandas

#Return boolean mask
missing = pices.isna()

#Interpolate linearly within missing windows
prices_interp = prices.interpolate('linear')

# Create a function we'll use to interpolate and plot
def interpolate_and_plot(prices, interpolation):

    # Create a boolean mask for missing values
    missing_values = prices.isna()

    # Interpolate the missing values
    prices_interp = prices.interpolate(interpolation)

    # Plot the results, highlighting the interpolated values in black
    fig, ax = plt.subplots(figsize=(10, 5))
    prices_interp.plot(color='k', alpha=.6, ax=ax, legend=False)

    # Now plot the interpolated values on top in red
    prices_interp[missing_values].plot(ax=ax, color='r', lw=3, legend=False)
    plt.show()


    # Your custom function
def percent_change(series):
    # Collect all *but* the last value of this window, then the final value
    previous_values = series[:-1]
    last_value = series[-1]

    # Calculate the % difference between the last value and the mean of earlier values
    percent_change = (last_value - np.mean(previous_values)) / np.mean(previous_values)
    return percent_change

# Apply your custom function and plot
prices_perc = prices.rolling(20).apply(percent_change)
prices_perc.loc["2014":"2015"].plot()
plt.show()


#Next we want to transform the data to standardizie the variance to detect and remove outliers
#After transformation, each point will represent the % change over a previous window
def percent_change(series):
    # Collect all *but* the last value of this window, then the final value
    previous_values = series[:-1]
    last_value = series[-1]

    # Calculate the % difference between the last value and the mean of earlier values
    percent_change = (last_value - np.mean(previous_values)) / np.mean(previous_values)
    return percent_change

# Apply your custom function and plot
brain_perc = brain.rolling(window=window_size).apply(percent_change)


plt.show()





Cross validation for fitting time series






'''