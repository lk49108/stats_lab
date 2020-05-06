import data_scheduler.lib_data_merger as mice_data
import feature_generator.FeatureExtraction as feature_generator
from tsfresh.feature_extraction.settings import EfficientFCParameters, ComprehensiveFCParameters
import matplotlib.pyplot as plt
import feature_visualization.visualization as plot
import classifier.LinearClassifier as LC
import pandas

if __name__=='__main__':
    mice_data_dir = r'C:\Users\marce\OneDrive - ETHZ\Education\ETH\Spring 2020\Statslab\Project_Neuroscience\dataset'
    md = mice_data.MiceDataMerger(mice_data_dir)
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
    #fc_parameters = EfficientFCParameters()
    #fc_parameters = ComprehensiveFCParameters()
    feature_generator = feature_generator.FeatureExtractor(fc_parameters, md, 'brain_signal', brain_half='right',
                                                           mouse_ids={165, 166}, slice_min=30, target="group",
                                                           part_last=10, equal_length= True)
    #relevant_features = feature_generator.relevantFeatures()
    extracted_features = feature_generator.getFeatures()
    #print(extracted_features)
    plot = plot.FeaturePlot(extracted_features, feature_generator)
    pl = plot.plotFeatures(plot_type=  'all') # either plot_type = 'all' or
                                                    # plot_type = ['subjectLineplot','overallLineplot','boxplot','histogram']
                                                    # with 1,2,3,4 arguments or alternatively one of these 4 as a string
    plt.show()
    #extracted_features = feature_generator.getFeatures(target_class= True)
    #pandas.set_option('display.max_rows', relevant_features.shape[0] + 1)
    #pandas.set_option('display.max_rows', extracted_features.shape[0] + 1)
    #pandas.set_option('display.max_columns', 760)
    #print(relevant_features)
    #print(extracted_features)
    #print(extracted_features.columns)
    #classifier = LC.LinearClassifier(relevant_features)
    #classifier = LC.LinearClassifier(extracted_features, penalty= 'l1')
    #classifier.classify(train_test = {'split_type': 'subject', 'train': [165], 'test' : [166]}) # splittype either
                                                            # 'subject' or 'ratio' with percentages of whole data
    #classifier.classify(train_test={'split_type': 'ratio', 'train': 0.7, 'test': 0.3})