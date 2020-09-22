import data_scheduler.lib_data_merger as mice_data
import feature_generator.FeatureExtraction as feature_generator
import matplotlib.pyplot as plt
import os

chunk_length = 15
mice_data_dir = r'C:\Users\lkokot\Desktop\ETHZ_STAT_MSC\sem_2\stats_lab\analysis\CSV data files for analysis'
out_dir = r'C:\Users\lkokot\Desktop\ETHZ_STAT_MSC\sem_2\stats_lab\analysis\feature_plots'
mice_ids = [126, 165, 166, 167, 170, 299, 302, 303, 306, 307, 323, 327]

model = {
    # 'spkt_welch_density' : None,@error
    'binned_entropy': [{'max_bins': 10}],
    'longest_strike_above_mean': None,
    'longest_strike_below_mean': None,
    # 'fft_coefficient' : None,@error
    'standard_deviation': None,
    'fft_aggregated': [{'aggtype': 'centroid'}, {'aggtype': 'variance'}, {'aggtype': 'skew'}, {'aggtype': 'kurtosis'}],
    'agg_autocorrelation': [{'f_agg': 'mean', 'maxlag': 40}, {'f_agg': 'median', 'maxlag': 40}, {'f_agg': 'var', 'maxlag': 40}],
}

def plot_multiple(df, file_ids, mouse_id):
    colors = {
        'sal' : 'red',
        'nea' : 'blue',
        'eth' : 'green',
        'glu' : 'brown'
    }

    ftrs = df.columns.to_list()
    ftrs.remove('time_min')
    ftrs.remove('chunk_id')

    for feat in ftrs:
        figure = plt.axes()

        for file_id in file_ids:
            indices = (df['chunk_id'] == file_id).tolist()
            plot_measurement_file(df.iloc[indices], y_col_name = feat, file_name = file_id, figure = figure, color = colors[file_id[1]])

        if '"' in feat:
            while '"' in feat:
                feat = feat[:feat.find('"')] + feat[feat.find('"')+1:]

        figure.legend()
        plt.title(feat)

        def make_dirs(filename):
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))

        plot_file = os.path.join(out_dir, feat, '{}_{}'.format(mouse_id, feat) + "." + 'png')
        make_dirs(plot_file)

        plt.savefig(plot_file)
        figure.clear()

def plot_measurement_file(df, time_col_name = 'time_min', y_col_name = None, file_name = None, figure = None, color = 'black'):
    if df is None:
        raise ValueError('Dataframe provided can not be None value')
    if time_col_name is None:
        raise ValueError('Time column name has to be provided')
    if file_name is None:
        raise ValueError('File name from which measurements were taken has to be provided')
    if figure is None:
        raise ValueError('Figure provided should not be of value None')
    if y_col_name is None:
        raise ValueError('Column name that corresponds to feature has to be provided to plot_measurement_function function')

    # df.index = df.index.to_series().apply(lambda x: int(x[x.find('-')+1:x.rfind('-')]))
    df = df.sort_values(by=['time_min'])
    df.plot(x = time_col_name, y = y_col_name, ax = figure, kind = 'line', label = str(file_name), c = color)

if __name__ == '__main__':
    print('start')
    md = mice_data.MiceDataMerger(mice_data_dir)
    for feat in model.items():
        for mouse_id in mice_ids:
            feature_vals = feature_generator.FeatureExtractor({feat[0] : feat[1]},
                                                                  md,
                                                                  'brain_signal',
                                                                  brain_half='right',
                                                                  mouse_ids={mouse_id},
                                                                  slices={
                                                                      'eth': (0, 100),
                                                                      'glu': (0, 100),
                                                                      'nea': (0, 100),
                                                                      'sal': (0, 100)
                                                                  },
                                                                  target='all_vs_all',
                                                                  part_last=chunk_length,
                                                                  equal_length=True,
                                                                  overlap_ratio=0.9)
            features = feature_vals.getFeatures(feature_dict=model)
            feature_time_df = features.drop(columns=['target_class'])
            plot_multiple(feature_time_df, feature_vals.mice_treat_file_ids, mouse_id)
