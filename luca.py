import data_scheduler.lib_data_merger as mice_data
import feature_generator.FeatureExtraction as feature_generator
from tsfresh.feature_extraction.settings import EfficientFCParameters, ComprehensiveFCParameters
import matplotlib.pyplot as plt
import feature_visualization.visualization as plot
import classifier.LinearClassifier as LC
import feature_visualization.visualization as plot

if __name__=='__main__':
#    mice_data_dir = r'C:\Users\lkokot\Desktop\ETHZ_STAT_MSC\sem_2\stats_lab\analysis\CSV data files for analysis'
    mice_data_dir = r'/Users/lucadisse/ETH/Master/FS20/StatsLab/CSV data files for analysis'
    md = mice_data.MiceDataMerger(mice_data_dir)

    # for defining own faetures fill them binto the dictionary
    own_fc_parameters = {
         'spkt_welch_density': [{'coeff': 2}, {'coeff': 5}, {'coeff': 8}],
        'longest_strike_above_mean': None,
        'longest_strike_below_mean': None,
        'fft_coefficient': [{'coeff': 45, 'attr': 'abs'}, {'coeff': 38, 'attr': 'abs'}, {'coeff': 35, 'attr': 'abs'},
                            {'coeff': 44, 'attr': 'abs'}, {'coeff': 52, 'attr': 'abs'}, {'coeff': 1, 'attr': 'abs'},
                            {'coeff': 41, 'attr': 'abs'}, {'coeff': 18, 'attr': 'abs'}, {'coeff': 58, 'attr': 'abs'},
                            {'coeff': 42, 'attr': 'abs'}, {'coeff': 36, 'attr': 'abs'}],
        "autocorrelation": [{"lag": 3}],
         "variance": None,
         "number_peaks": [{"n": 5}],
    }

    fc_parameters = EfficientFCParameters()

    #set target to treatment to check if we actually predict the nea vs all correctly
    #skipping id 168, 170 they said its messed up data for the brain
    #TODO check all mice for messed up data, maybe in R check all histograms.
    mice_ids = {126, 165, 166, 167, 299, 302, 303, 306, 307, 323, 327}
    validation_mice = {165, 307, 327}
    training_mice = {166, 170, 299, 306}

    #target arguments: use 'nea_vs_all' or 'all_vs_all'
    #for few training mice set part_last to 10 mins, for all training mice, set part last to 20
    #TODO only to feature selection over the training data and remember the names
    #TODO read features from saved format

    # as signal choose either the running or brain_signal
    train_feature_generator = feature_generator.FeatureExtractor(md, signal_type='brain_signal', brain_half='right',
                                                           mouse_ids=mice_ids-validation_mice, slice_min=35, target='all_vs_all',
                                                           part_last=15, slice_end = 55, overlap_ratio = 0.9)

    test_feature_generator = feature_generator.FeatureExtractor(md, signal_type='brain_signal', brain_half='right',
                                                           mouse_ids=validation_mice, slice_min=35, target='all_vs_all',
                                                           part_last=15, slice_end = 55, overlap_ratio = 0.9)

    relevant_train_features, relevant_fc_parameters = train_feature_generator.relevantFeatures(feature_dict=fc_parameters)

    feature_relevance = train_feature_generator.getRelevance(features=relevant_train_features.drop('target_class', axis=1),
                                                             target=relevant_train_features['target_class'])

    feature_relevance.sort_values(by=['p_value']).to_csv('feature_relevance.csv')

    print(relevant_fc_parameters)

    relevant_test_features = test_feature_generator.getFeatures(feature_dict=relevant_fc_parameters)

    feature_block = relevant_train_features.append(relevant_test_features)
    #feature_generator = feature_generator.FeatureExtractor(own_fc_parameters, md, 'brain_signal', brain_half='right',
    #                                                       mouse_ids=training_mice, slice_min=30, target='all_vs_all',
    #                                                       part_last=15, equal_length= True, slice_end = 60, overlap_ratio = 0.8)

    #validation features
    print('Feature Values:\n', feature_block)

    #print(feature_block.columns)
    classifier = LC.LinearClassifier(feature_block,model_type='svm', C_val = 1000)

    # splittype either 'subject' or 'ratio' with percentages of whole data
    classifier.classify(train_test={'split_type': 'subject', 'train': list(mice_ids-validation_mice), 'test' : list(validation_mice)})
    classifier.classify(train_test={'split_type': 'ratio', 'train': 0.7, 'test': 0.3})
