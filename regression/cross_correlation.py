#In case the lib_data_merger library is not recognized, first insert the statement below in the python console
#This should change your directory to /stats_lab (where you have subfolders like data_scheduler, ...)
#And then you insert this directory in the system path, so that when the modules are loaded, the data_scheduler folder is recognized
'''
import os as os
os.chdir('stats_lab')
sys.path.insert(0, os.getcwd())
'''

import data_scheduler.lib_data_merger as mice_data
import pandas as pd
from numpy.lib.stride_tricks import as_strided
from numpy.lib import pad
import numpy as np
import matplotlib.pyplot as plt

#Anova libraries (use pip install pingouin in Anaconda prompt)
import pingouin as pg

#Different approach for ANOVA
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scipy.stats as stats
from statsmodels.graphics.gofplots import qqplot

#First we want to fetch the relevant data which includes the brain and running activity
mice_data_dir = r'C:\Users\Massimo\Documents\StatsLab\New Data' #indicate here the directory where your .csv files of mouse data lie (use "r" in front of path in case you use "\" for path)
md = mice_data.MiceDataMerger(mice_data_dir)
treatments = ['glu', 'eth', 'sal', 'nea']
signals = ['brain_signal', 'running', 'v_o2', 'v_co2', 'rq', 'heat']
df_anova_r = pd.DataFrame(columns = ['R','Treatment'])
df_anova_lag = pd.DataFrame(columns = ['Lag','Treatment'])
window = 200


#Functions
def crosscorr(datax, datay, method, lag=0, wrap=False):
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
        return datax.corr(shiftedy, method = method)
    else:
        return datax.corr(datay.shift(lag), method = method)
def plot_crosscorr(brain_signal, running_signal, min_lowerbound, min_upperbound, method):
    brain = brain_signal.sliced_data(min_lowerbound).left_slice(min_upperbound).get_pandas()
    brain = brain.set_index('time_min')
    running = running_signal.sliced_data(min_lowerbound).left_slice(min_upperbound).get_pandas()
    running = running.set_index('time_min')

    d1 = brain.iloc[:, hemisphere]
    d2 = running.iloc[:, hemisphere]
    shift = 30
    rs = [crosscorr(d1, d2, method, lag) for lag in range(-int(shift), int(shift))]
    lag = np.ceil(len(rs) / 2) - np.argmax(rs)
    f, ax = plt.subplots(figsize=(14, 3))
    ax.plot(range(-int(shift), int(shift)), rs)
    ax.axvline(np.ceil(len(rs))-2*shift, color='k', linestyle='--', label='Center')
    ax.axvline(np.argmax(rs)-shift, color='r', linestyle='--', label='Peak synchrony')
    ax.set(title=f'Lag = {lag} shift\n Brain signal leads <> Running signal leads \n' + str(mouse_ids[j]) + str(treatments[i]),
           xlabel='Lag',
           ylabel= corr_method + ' correlation coefficient')

    plt.legend()
def return_crosscorr(brain_signal, running_signal, min_lowerbound, min_upperbound, method):
    brain = brain_signal.sliced_data(min_lowerbound).left_slice(min_upperbound).get_pandas()
    brain = brain.set_index('time_min')
    running = running_signal.sliced_data(min_lowerbound).left_slice(min_upperbound).get_pandas()
    running = running.set_index('time_min')

    d1 = brain.iloc[:, hemisphere]
    d2 = running.iloc[:, hemisphere]
    shift = 30
    rs = [crosscorr(d1, d2, method, lag) for lag in range(-int(shift), int(shift))]
    return np.max(rs)
def return_lag(brain_signal, running_signal, min_lowerbound, min_upperbound, method):
    brain = brain_signal.sliced_data(min_lowerbound).left_slice(min_upperbound).get_pandas()
    brain = brain.set_index('time_min')
    running = running_signal.sliced_data(min_lowerbound).left_slice(min_upperbound).get_pandas()
    running = running.set_index('time_min')

    d1 = brain.iloc[:, hemisphere]
    d2 = running.iloc[:, hemisphere]
    shift = 30
    rs = [crosscorr(d1, d2, method, lag) for lag in range(-int(shift), int(shift))]
    lag = np.ceil(len(rs) / 2) - np.argmax(rs)
    return lag
def rolling_correlation_plot(brain_signal, running_signal,  min_lowerbound, min_upperbound, window_size = window):
    brain = brain_signal.sliced_data(min_lowerbound).left_slice(min_upperbound).get_pandas()
    brain = brain.set_index('time_min')
    running = running_signal.sliced_data(min_lowerbound).left_slice(min_upperbound).get_pandas()
    running = running.set_index('time_min')

    d1 = brain.iloc[:, hemisphere]
    d2 = running.iloc[:, hemisphere]


    # Compute rolling window synchrony
    rolling_r = pd.Series(rolling_spearman(d1, d2, window_size))
    f,ax=plt.subplots(2,1,figsize=(14,6),sharex=True)
    d1.rolling(window=30,center=True).median().plot(ax=ax[0])
    d2.rolling(window=30,center=True).median().plot(ax=ax[0])
    ax[0].set(xlabel='Minutes',ylabel='Brain/running signal (Rolling median)')
    rolling_r.plot(ax=ax[1])
    ax[1].set(xlabel='Minutes',ylabel='Spearman correlation coefficient')
    plt.suptitle('Brain/running signal and rolling window correlation for ' + str(mouse_ids[j]) + str(treatments[i]))
def rolling_spearman(seqa, seqb, window):
    stridea = seqa.strides[0]
    ssa = as_strided(seqa, shape=[len(seqa) - window + 1, window], strides=[stridea, stridea])
    strideb = seqa.strides[0]
    ssb = as_strided(seqb, shape=[len(seqb) - window + 1, window], strides =[strideb, strideb])
    ar = pd.DataFrame(ssa)
    br = pd.DataFrame(ssb)
    ar = ar.rank(1)
    br = br.rank(1)
    corrs = ar.corrwith(br, 1)
    output = pd.Series(pad(corrs, (window - 1, 0), 'constant', constant_values=np.nan))
    output.index = seqa.index
    return output

#Interval in minutes (this is the time interval selected for all brain and running signals)
lower = 30
upper = 90


#Options:
hemisphere = 0 #put 0 for left brain hemisphere, put 1 for right brain hemisphere
option_plot_crosscorr = False  #set True if you want to plot the Cross-Correlation plot for each subject/treatment combination
option_moving_correlation = True  #set True if you want to plot the moving correlation
#Attention: Calculating the moving correlation does take quite some time

window = 200 #for moving correlation plot
corr_method = 'spearman' #pearson for linear relationship; spearman for monotone relationship
option_anova = False #set True if you want to conduct a one-way ANOVA to compare means of treatments
option_model_assumptions = False #set true if you want to test ANOVA model assumptions (option_anova must also be true)
option_mutiple_comparison = False #set true if you want to do a multiple comparison (compare all different treatment pairs)
option_boxplots = True #set true if you want to plot the boxplot for the different treatment means

#Select which mice you want to loop through
mouse_ids_list = [126, 165, 166, 167, 168, 170, 176, 218, 223, 302, 303, 306, 307, 327]
mouse_ids = mouse_ids_list[0:1] #(to select all mice use [0:14])


#We loop through all mice and treatments combinations
#We use a nested structure to cover the following cases:
#If there is no data available, the loop returns a None argument, which leads to an error
#If there are multiple subjects (e.g. multiple observations for 165) then the loop returns a list
for j in range(len(mouse_ids)):
    for i in range(len(treatments)):
        brain_signal = md.fetch_mouse_signal(mouse_id = mouse_ids[j], treat=treatments[i], signal='brain_signal')
        running_signal = md.fetch_mouse_signal(mouse_id = mouse_ids[j], treat=treatments[i], signal='running')

        if option_plot_crosscorr:
            # Visualization of data
            if brain_signal is not None and running_signal is not None:
                if type(brain_signal) is not list and type(running_signal) is not list:
                    plot_crosscorr(brain_signal, running_signal, lower, upper, method=corr_method)

                elif type(brain_signal) is list and type(running_signal) is list:
                    for m in range(len(brain_signal)):
                        plot_crosscorr(brain_signal[m], running_signal[m], lower, upper, method=corr_method)


        # Output of optimal Pearson r value and lag
        if brain_signal is not None and running_signal is not None:
            if type(brain_signal) is not list and type(running_signal) is not list:
                df_anova_r = df_anova_r.append(pd.DataFrame({'R': [return_crosscorr(brain_signal, running_signal, lower, upper, method=corr_method)],
                                              'Treatment': [treatments[i]]}))
                df_anova_lag = df_anova_lag.append(pd.DataFrame({'Lag': [return_lag(brain_signal, running_signal, lower, upper, method=corr_method)],
                                                             'Treatment': [treatments[i]]}))


            elif type(brain_signal) is list and type(running_signal) is list:
                for m in range(len(brain_signal)):
                    df_anova_r = df_anova_r.append(pd.DataFrame({'R': [return_crosscorr(brain_signal[m], running_signal[m], lower, upper, method=corr_method)],
                                                  'Treatment': [treatments[i]]}))
                    df_anova_lag = df_anova_lag.append(pd.DataFrame({'Lag': [return_lag(brain_signal[m], running_signal[m], lower, upper, method=corr_method)],
                                      'Treatment': [treatments[i]]}))

        else:
            print(mouse_ids[j], treatments[i]) #we print all the subject/treatment combinations that are not covered

        if option_moving_correlation:
            # Visualization of data
            if brain_signal is not None and running_signal is not None:
                if type(brain_signal) is not list and type(running_signal) is not list:
                    rolling_correlation_plot(brain_signal, running_signal, lower, upper, window_size = window)

                elif type(brain_signal) is list and type(running_signal) is list:
                    for m in range(len(brain_signal)):
                        rolling_correlation_plot(brain_signal[m], running_signal[m], lower, upper, window_size = window)

if option_anova:
    # We perform a simple one-way ANOVA for optimal R values and Lag
    model_r = ols('R ~ C(Treatment)', data=df_anova_r).fit()
    anova_table_r = sm.stats.anova_lm(model_r, typ=2)
    print(anova_table_r)

    model_lag = ols('Lag ~ C(Treatment)', data=df_anova_lag).fit()
    anova_table_lag = sm.stats.anova_lm(model_lag, typ=2)
    print(anova_table_lag)

    if option_model_assumptions:
        # Test ANOVA model assumptions (normality and homoscedasticity)
        print('Normality test (if pvalue is below 0.05 then we can reject the null hypothesis that the data is normally distributed)')
        w, pvalue = stats.shapiro(model_r.resid)
        print(pvalue, "normality test for optimal correlation correlation coefficient residuals")

        w, pvalue = stats.shapiro(model_lag.resid)
        print(pvalue, "normality test for optimal lag residuals")

        # QQ Plot
        qqplot(df_anova_r['R'], line='s')
        plt.title(label='QQ Plot Residuals Optimal Correlation Coefficient')
        plt.show()


        qqplot(df_anova_lag['Lag'], line='s')
        plt.title(label='QQ Plot Residuals Optimal Lag')
        plt.show()

        # Tukey Anscombe Plot
        plt.figure()
        plt.scatter(model_r.fittedvalues, model_r.resid)
        plt.title(label='Residuals vs. Fitted Values for Optimal Correlation Coefficient')
        plt.ylabel(ylabel='Residuals')
        plt.xlabel(xlabel='Fitted values')
        plt.hlines(y=0, xmin=min(model_r.fittedvalues), xmax=max(model_r.fittedvalues),
                   linestyles='dashed')

        plt.figure()
        plt.scatter(model_lag.fittedvalues, model_lag.resid)
        plt.title(label='Residuals vs. Fitted Values for Optimal Lag')
        plt.ylabel(ylabel='Residuals')
        plt.xlabel(xlabel='Fitted values')
        plt.hlines(y=0, xmin=min(model_lag.fittedvalues), xmax=max(model_lag.fittedvalues), linestyles='dashed')


if option_mutiple_comparison:
    # Multiple pairwise comparison (Tukey HSD)
    #It has been proven to be conservative for one-way ANOVA with unequal sample sizes
    m_comp = pairwise_tukeyhsd(endog=df_anova_r['R'], groups=df_anova_r['Treatment'], alpha=0.05)
    print(m_comp)

    #In addition, in case of unequal variances  we perform a Games-Howell Test
    gh = pg.pairwise_gameshowell(data=df_anova_r, dv='R', between='Treatment')
    print(gh)



if option_boxplots:
    #We visualize the results for boxplots
    df_anova_r.boxplot(column='R', by='Treatment', showmeans=True)
    plt.title(label='Boxplot for ' + corr_method + ' correlation coefficient across treatments')
    plt.ylabel(ylabel= ('Optimal ' + corr_method + ' correlation coefficient'))

    df_anova_lag.boxplot(column='Lag', by='Treatment', showmeans=True)
    plt.title(label='Boxplot for optimal lag across treatments')
    plt.ylabel(ylabel='Lag (in time period of time series)')
    plt.show()