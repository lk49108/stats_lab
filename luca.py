import data_scheduler.lib_data_merger as mice_data
import feature_generator.FeatureExtraction as feature_generator
from tsfresh.feature_extraction.settings import EfficientFCParameters, ComprehensiveFCParameters
import matplotlib.pyplot as plt
import feature_visualization.visualization as plot
import classifier.LinearClassifier as LC
import feature_visualization.visualization as plot

if __name__=='__main__':
    mice_data_dir = r'C:\Users\lkokot\Desktop\ETHZ_STAT_MSC\sem_2\stats_lab\analysis\CSV data files for analysis'
    md = mice_data.MiceDataMerger(mice_data_dir)
    md2 = mice_data.MiceDataMerger(mice_data_dir)

    # for defining own features fill into the
    own_fc_parameters = {
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

    fc_parameters = EfficientFCParameters()

    #set target to treatment to check if we actually predict the nea vs all correctly
    #skipping id 168, they said its messed up data for the brain
    #TODO check all mice for messed up data, maybe in R check all histograms.
    mice_ids = {126, 165, 166, 167, 170, 299, 302, 303, 306, 307, 323, 327}
    validation_mice = {165, 170, 307, 327}
    training_mice = {166, 170, 299, 306}

    #target arguments: use 'nea_vs_all' or 'all_vs_all'
    #for few training mice set part_last to 10 mins, for all training mice, set part last to 20
    #TODO only to feature selection over the training data and remember the names
    #TODO read features from saved format

    # as signal choose either the running or brain_signal
    feature_generator = feature_generator.FeatureExtractor(own_fc_parameters, md, 'brain_signal', brain_half='right',
                                                           mouse_ids=training_mice, slice_min=30, target='all_vs_all',
                                                           part_last=15, equal_length= True, slice_end = 60, overlap_ratio = 0.8)

    relevant_features = feature_generator.relevantFeatures()
    #validation features
    print('Number of features selected: ', relevant_features)


    classifier = LC.LinearClassifier(relevant_features, C_val = 1)
    #classifier = LC.LinearClassifier(extracted_features)
    # splittype either 'subject' or 'ratio' with percentages of whole data
    classifier.classify(train_test={'split_type': 'subject', 'train': list(mice_ids-validation_mice), 'test' : list(validation_mice)})
    classifier.classify(train_test={'split_type': 'ratio', 'train': 0.7, 'test': 0.3})
