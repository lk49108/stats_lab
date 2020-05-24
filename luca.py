import data_scheduler.lib_data_merger as mice_data
import feature_generator.FeatureExtraction as feature_generator
from tsfresh.feature_extraction.settings import EfficientFCParameters, ComprehensiveFCParameters
import classifier.LinearClassifier as LC
import pandas as pd

if __name__=='__main__':
    mice_data_dir = r'/Users/lucadisse/ETH/Master/FS20/StatsLab/joint_data'
    md = mice_data.MiceDataMerger(mice_data_dir)

    nea_mice = {167, 168, 176, 218, 302, 303, 306, 307, 327}
    glu_mice = {166, 167, 170, 306, 307}
    sal_mice = {126, 165, 166, 167,168,176,218,306}
    eth_mice = {165,166,168,218,223,306,307}

    fc_parameters = EfficientFCParameters()

    mice_ids = {126, 165, 166, 167, 168, 170, 176, 218, 223, 302, 303, 306, 307, 327}

    #sal vs nea validation
    #validation_mice = {167, 168, 176}
    #ethanol vs saline
    #validation_mice = {218, 223, 306}
    #sal vs all
    #validation_mice = {166,306,307}
    #all vs all
    validation_mice = {166, 167, 307}

    #sal_vs_all
    #eth_vs_sal
    #glu_vs_sal
    #nea_vs_sal
    target = "all_vs_all"
    overlap = 0.9
    train_feature_generator = feature_generator.FeatureExtractor(md, signal_type='brain_signal', brain_half='left',
                                                           mouse_ids=mice_ids, slice_min=35, target=target,
                                                           part_last=15, slice_end = 55, overlap_ratio = overlap)

    test_feature_generator = feature_generator.FeatureExtractor(md, signal_type='brain_signal', brain_half='left',
                                                           mouse_ids=validation_mice, slice_min=35, target=target,
                                                           part_last=15, slice_end = 55, overlap_ratio = overlap)

    recompute_features = True
    if recompute_features:
        relevant_train_features, relevant_fc_parameters = train_feature_generator.relevantFeatures(feature_dict=fc_parameters)

        feature_relevance = train_feature_generator.getRelevance(features=relevant_train_features.drop('target_class', axis=1),
                                                                 target=relevant_train_features['target_class'])

        feature_relevance.sort_values(by=['p_value']).to_csv('feature_relevance.csv')

        #print(relevant_fc_parameters)

        relevant_test_features = test_feature_generator.getFeatures(feature_dict=relevant_fc_parameters)

        feature_block = relevant_train_features.append(relevant_test_features)
        feature_block.to_csv('features.csv', index=False, header=False)
    else:
        feature_block = pd.read_csv('features.csv', header=True, index=True)

    #validation features
    print('Feature Values:\n', feature_block)

    #print(feature_block.columns)
    classifier = LC.LinearClassifier(feature_block, model_type='logistic', C_val = 1)

    # splittype either 'subject' or 'ratio' with percentages of whole data
    classifier.classify(train_test={'split_type': 'subject', 'train': list(mice_ids-validation_mice), 'test' : list(validation_mice)})
    classifier.classify(train_test={'split_type': 'ratio', 'train': 0.7, 'test': 0.3})
