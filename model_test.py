import data_scheduler.lib_data_merger as mice_data
import feature_generator.FeatureExtraction as feature_generator
from tsfresh.feature_extraction.settings import EfficientFCParameters, ComprehensiveFCParameters
import classifier.LinearClassifier as LC

chunk_length = 17
out_file = r'C:\Users\lkokot\Desktop\ETHZ_STAT_MSC\sem_2\stats_lab\analysis\log.out'
mice_data_dir = r'C:\Users\lkokot\Desktop\ETHZ_STAT_MSC\sem_2\stats_lab\analysis\CSV data files for analysis'
mice_ids = {126, 165, 166, 167, 170, 299, 302, 303, 306, 307, 323, 327}
model = {
    'binned_entropy': [{'max_bins': 10}],
    'longest_strike_above_mean': None,
    'standard_deviation': None,
    'variance_larger_than_standard_deviation': None,
    'fft_aggregated': [{'aggtype': 'centroid'}, {'aggtype': 'variance'}, {'aggtype': 'skew'}, {'aggtype': 'kurtosis'}],
    'longest_strike_below_mean': None,
    'agg_autocorrelation': [{'f_agg': 'mean', 'maxlag': 40}, {'f_agg': 'median', 'maxlag': 40}, {'f_agg': 'var', 'maxlag': 40}],
    'minimum' : None
}

if __name__ == '__main__':
    with open(out_file, 'w') as out:
        md = mice_data.MiceDataMerger(mice_data_dir)
        all_features = EfficientFCParameters()
        feature_vals = feature_generator.FeatureExtractor(all_features,
                                                          md,
                                                          'brain_signal',
                                                          brain_half='right',
                                                          mouse_ids=mice_ids,
                                                          slices={
                                                              'eth': (36, 60),
                                                              'glu': (31, 55),
                                                              'nea': (36, 60),
                                                              'sal': (31, 55)
                                                          },
                                                          target='all_vs_all',
                                                          part_last=chunk_length,
                                                          equal_length=True,
                                                          overlap_ratio=0.9)
        features = feature_vals.getFeatures(feature_dict=model)
        classifier = LC.LinearClassifier(features, C_val=1, model_type='svm')
        acc, conf_matr = classifier.k_fold_train_classifier(mice_treat_file_ids=feature_vals.mice_treat_file_ids, log=out)

        out.write('Chunk length: ' + str(chunk_length) + '\n')
        out.write('\n\nClassification accuracy:\n')
        out.write(str(acc) + '\n')
        out.write('Confusion matrix:\n')
        out.write(str(conf_matr) + '\n')
        out.write('Features:\n')
        out.write(str(model) + '\n')