'''
import os as os
os.chdir('stats_lab')
sys.path.insert(0, os.getcwd())
'''

import data_scheduler.lib_data_merger as mice_data
import pandas as pd
import matplotlib.pyplot as plt
import os as os
import numpy as np
import matplotlib.pyplot as plt

#Anova libraries (use pip install pingouin in Anaconda prompt)
import pingouin as pg


#First we want to fetch the relevant data which includes the brain and running activity
mice_data_dir = r'C:\Users\Massimo\Documents\StatsLab\New Data'
md = mice_data.MiceDataMerger(mice_data_dir)
treatments = ['glu', 'eth', 'sal', 'nea']
signals = ['brain_signal', 'running', 'v_o2', 'v_co2', 'rq', 'heat']
df_anova_r = pd.DataFrame(columns = ['R','Treatment'])
df_anova_off = pd.DataFrame(columns = ['Offset','Treatment'])

mouse_ids_list = [165, 166, 167, 126, 170, 299, 302, 303, 306, 307, 323, 327]
mouse_ids = mouse_ids_list[0:12] #(to select all mice use [0:12])

def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation.
    Shifted data filled with NaNs

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else:
        return datax.corr(datay.shift(lag))
def plot_crosscorr(brain_signal, running_signal, min_lowerbound, min_upperbound,):
    brain = brain_signal.sliced_data(min_lowerbound).left_slice(min_upperbound).get_pandas()
    brain = brain.set_index('time_min')
    running = running_signal.sliced_data(min_lowerbound).left_slice(min_upperbound).get_pandas()
    running = running.set_index('time_min')

    d1 = brain.iloc[:, 1]
    d2 = running.iloc[:, 0]
    minutes = 30
    rs = [crosscorr(d1, d2, lag) for lag in range(-int(minutes), int(minutes))]
    offset = np.ceil(len(rs) / 2) - np.argmax(rs)
    f, ax = plt.subplots(figsize=(14, 3))
    ax.plot(rs)
    ax.axvline(np.ceil(len(rs) / 2), color='k', linestyle='--', label='Center')
    ax.axvline(np.argmax(rs), color='r', linestyle='--', label='Peak synchrony')
    ax.set(title=f'Offset = {offset} minutes\n Brain signal leads <> Running signal leads \n + str(mouse_ids[j]) + str(treatments[i])',
           xlabel='Offset',
           ylabel='Pearson r')
    plt.legend()
def return_crosscorr(brain_signal, running_signal, min_lowerbound, min_upperbound):
    brain = brain_signal.sliced_data(min_lowerbound).left_slice(min_upperbound).get_pandas()
    brain = brain.set_index('time_min')
    running = running_signal.sliced_data(min_lowerbound).left_slice(min_upperbound).get_pandas()
    running = running.set_index('time_min')

    d1 = brain.iloc[:, 1]
    d2 = running.iloc[:, 0]
    minutes = 30
    rs = [crosscorr(d1, d2, lag) for lag in range(-int(minutes), int(minutes))]
    return np.max(rs)
def return_offset(brain_signal, running_signal, min_lowerbound, min_upperbound):
    brain = brain_signal.sliced_data(min_lowerbound).left_slice(min_upperbound).get_pandas()
    brain = brain.set_index('time_min')
    running = running_signal.sliced_data(min_lowerbound).left_slice(min_upperbound).get_pandas()
    running = running.set_index('time_min')

    d1 = brain.iloc[:, 1]
    d2 = running.iloc[:, 0]
    minutes = 30
    rs = [crosscorr(d1, d2, lag) for lag in range(-int(minutes), int(minutes))]
    offset = np.ceil(len(rs) / 2) - np.argmax(rs)
    return offset

#Interval in minutes
lower = 30
upper = 70

#We loop through all mice and treatments to get two columns: Pearson R and the respective treatment
for j in range(len(mouse_ids)):
    for i in range(len(treatments)):
        brain_signal = md.fetch_mouse_signal(mouse_id = mouse_ids[j], treat=treatments[i], signal='brain_signal')
        running_signal = md.fetch_mouse_signal(mouse_id = mouse_ids[j], treat=treatments[i], signal='running')
        '''
        # Visualization of data
        if brain_signal is not None and running_signal is not None:
            if type(brain_signal) is not list and type(running_signal) is not list:
                plot_crosscorr(brain_signal, running_signal)

            elif type(brain_signal) is list and type(running_signal) is list:
                for m in range(len(brain_signal)):
                    plot_crosscorr(brain_signal[m], running_signal[m])
        '''

        # Output of optimal Pearson r value and offset
        if brain_signal is not None and running_signal is not None:
            if type(brain_signal) is not list and type(running_signal) is not list:
                df_anova_r = df_anova_r.append(pd.DataFrame({'R': [return_crosscorr(brain_signal, running_signal, lower, upper)],
                                              'Treatment': [treatments[i]]}))
                df_anova_off = df_anova_off.append(pd.DataFrame({'Offset': [return_offset(brain_signal, running_signal, lower, upper)],
                                                             'Treatment': [treatments[i]]}))


            elif type(brain_signal) is list and type(running_signal) is list:
                for m in range(len(brain_signal)):
                    df_anova_r = df_anova_r.append(pd.DataFrame({'R': [return_crosscorr(brain_signal[m], running_signal[m], lower, upper)],
                                                  'Treatment': [treatments[i]]}))
                    df_anova_off = df_anova_off.append(pd.DataFrame({'Offset': [return_offset(brain_signal[m], running_signal[m], lower, upper)],
                                      'Treatment': [treatments[i]]}))

        else:
            print(mouse_ids[j], treatments[i])

#We perform a simple one-way ANOVA for optimal R values
aov_r = pg.anova(data=df_anova_r, dv='R', between='Treatment', ss_type=2, detailed=True)
print(aov_r)
aov_off = pg.anova(data=df_anova_off, dv='Offset', between='Treatment', ss_type=2, detailed=True)
print(aov_off)


#We visualize the results for boxplots
df_anova_r.boxplot(column='R', by='Treatment')
plt.title(label='Boxplot for Pearson R across treatments')

df_anova_off.boxplot(column='Offset', by='Treatment')
plt.title(label='Boxplot for Offset across treatments')
plt.show()





