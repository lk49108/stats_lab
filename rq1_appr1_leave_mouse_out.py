# RQ1 Approach 1
# This file was used to compute the accuracy table seen in the final presentation
# Here we perform a leave-one-mouse-out cross-validation

import data_scheduler.lib_data_merger as mice_data
import feature_generator.FeatureExtraction as feature_generator
from tsfresh.feature_extraction.settings import EfficientFCParameters, ComprehensiveFCParameters
import classifier.LinearClassifier as LC
import numpy as np
import pandas as pd

if __name__=='__main__':
    #change path!
    mice_data_dir = r'/Users/lucadisse/ETH/Master/FS20/StatsLab/exclude'
    md = mice_data.MiceDataMerger(mice_data_dir)

    # we can only consider mice sets that actually contain the desired treatments
    # here we removed measurements that contained the same treatment a mice mulitiple times
    nea_mice = {165, 166, 167, 168, 176, 218, 299, 302, 303, 306, 307, 327}
    glu_mice = {165, 166, 167, 168, 170, 306, 307}
    sal_mice = {126, 165, 166, 167, 168, 170, 176, 218, 302, 303, 306, 323}
    eth_mice = {165, 166, 167, 168, 170, 218, 223, 303, 306, 307}
    mice_ids = {126, 165, 166, 167, 168, 170, 176, 218, 223, 302, 303, 306, 307, 327}

    mice_sets = [sal_mice, eth_mice, glu_mice, nea_mice, mice_ids]

    fc_parameters = EfficientFCParameters()

    targets = ['sal_vs_all', 'eth_vs_sal', 'glu_vs_sal', 'nea_vs_sal', 'all_vs_all']

    acc_vals = np.zeros(shape=(len(targets), len(mice_ids)))
    overlap = 0.9

    target_index = 0
    for i in targets:
        mice_set_index = 0
        for j in list(mice_sets)[target_index]:
            train_feature_generator = feature_generator.FeatureExtractor(md, signal_type='brain_signal', brain_half='left',
                                                                   mouse_ids=mice_sets[target_index]-{j}, slice_min=35, target=i,
                                                                   part_last=15, slice_end = 90, overlap_ratio = overlap)

            test_feature_generator = feature_generator.FeatureExtractor(md, signal_type='brain_signal', brain_half='left',
                                                                   mouse_ids={j}, slice_min=35, target=i,
                                                                   part_last=15, slice_end = 90, overlap_ratio = overlap)

            relevant_train_features, relevant_fc_parameters = train_feature_generator.relevantFeatures(feature_dict=fc_parameters)

            feature_relevance = train_feature_generator.getRelevance(features=relevant_train_features.drop('target_class', axis=1),
                                                                     target=relevant_train_features['target_class'])

            relevant_test_features = test_feature_generator.getFeatures(feature_dict=relevant_fc_parameters)

            feature_block = relevant_train_features.append(relevant_test_features)

            classifier = LC.LinearClassifier(feature_block,model_type='logistic', C_val = 1)
            acc_score  = classifier.classify(train_test={'split_type': 'subject', 'train': list(mice_sets[target_index]-{j}), 'test' : j})

            acc_vals[target_index,mice_set_index] = acc_score / len(mice_sets[target_index])

            mice_set_index = mice_set_index + 1
        target_index = target_index + 1


    all_scores = np.sum(acc_vals, axis=1)
    print('Average accuracy scores: ', all_scores)