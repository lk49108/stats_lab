import data_scheduler.lib_data_merger as mice_data
import feature_generator.FeatureExtraction as feature_generator
import matplotlib.pyplot as plt
import feature_visualization.visualization as plot

if __name__=='__main__':
    # mice_data_dir = r'C:\Users\marce\OneDrive - ETHZ\Education\ETH\Spring 2020\Statslab\Project_Neuroscience\dataset'
    # md = mice_data.MiceDataMerger(mice_data_dir)
    # # feature_data = featureDataPreparation(md, chunk_duration=10, signal_type='brain_signal')
    # plotFeatures(feature_data)
    # data_eth_run = md.fetch_mouse_signal(166, 'eth', 'running')
    # #print(data_eth_run.get_pandas())
    # # data_eth_run = data_eth_run.sliced_data(time=True, slice_min=30)
    # #
    # #
    # # #data_glu_run = md.fetch_mouse_signal(165, 'sal', 'running')
    # # #data_sal_run = md.fetch_mouse_signal(165, 'nea', 'running')
    # #
    # # #features = getfewFeatures(data_eth_run, signal="running", print_feat=True)
    # # #print(features.columns)
    # #
    # # #plotFeatures(md)
    # #
    # # features = getallFeatures(md, signal='running', slice_min=30)
    # # print(features)
    # # #features[features['treatment'] == "glu"].plot(subplots=True, sharex=True, figsize=(10, 10))
    # # #plt.show()
    # # #print(features.columns)
    #
    # signal = md.fetch_mouse_signal(166, 'eth', 'running')
    # signal = signal.sliced_data(29.997)
    # #print(signal.get_pandas(time=True))
    # #print(signal.chunks)
    # chunks = signal.partition_data(part_last=5)
    # print('num of chunks:', len(chunks))
    # for chunk in chunks:
    #     df = chunk.get_pandas(time=True)
    #     print(df['time_min'].min(), df['time_min'].max(), chunk.chunks)
    fc_parameters = {
        "large_standard_deviation": [{"r": 0.05}, {"r": 0.1}],
        "abs_energy": None,
        "autocorrelation": [{"lag": 3}],
        "variance": None,
        "number_peaks": [{"n": 10}],
        "count_above_mean": None,
        "longest_strike_below_mean": None,
        "mean": None,
        "maximum": None,
        "median": None,
        "variance": None
    }

    #for binary classification set target to "treatment"
    #any other string will lead to multiclass classification


    #print(relevant_features)

    #classifier = LC.LinearClassifier(relevant_features)
    #classifier.classify()
