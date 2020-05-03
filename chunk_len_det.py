import data_scheduler.lib_data_merger as mice_data
import feature_generator.FeatureExtraction as feature_generator
from tsfresh.feature_extraction.settings import EfficientFCParameters, ComprehensiveFCParameters
import classifier.LinearClassifier as LC

mice_data_dir = r'C:\Users\lkokot\Desktop\ETHZ_STAT_MSC\sem_2\stats_lab\analysis\CSV data files for analysis'
out_file = r'C:\Users\lkokot\Desktop\ETHZ_STAT_MSC\sem_2\stats_lab\analysis\log.out'

mice_ids = {126, 165, 166, 167, 170, 299, 302, 303, 306, 307, 323, 327}
mice_ids_clean = {126, 302, 303, 323}
validation_mice = {165, 170, 307, 327}
training_mice = {166, 170, 299, 306}
own_fc_parameters = {'binned_entropy': [{'max_bins': 10}],
                     'longest_strike_above_mean': None,
                     'standard_deviation': None,
                     'variance_larger_than_standard_deviation': None,
                     'fft_aggregated': [{'aggtype': 'centroid'}, {'aggtype': 'variance'}, {'aggtype': 'skew'}, {'aggtype': 'kurtosis'}],
                     'longest_strike_below_mean': None
                     }

if __name__=='__main__':
    md = mice_data.MiceDataMerger(mice_data_dir)

    features = EfficientFCParameters()

    with open(out_file, 'w') as out:

        for chunk_length in range(15, 16):
            try:
                print('Chunk length:', chunk_length)
                feature_vals = feature_generator.FeatureExtractor(own_fc_parameters,
                                                                  md,
                                                                  'brain_signal',
                                                                  brain_half='right',
                                                                  mouse_ids=mice_ids,
                                                                  slices={
                                                                      'eth': (36, 55),
                                                                      'glu': (31, 50),
                                                                      'nea': (36, 55),
                                                                      'sal': (31, 50)
                                                                  },
                                                                  target='all_vs_all',
                                                                  part_last=chunk_length,
                                                                  equal_length=True,
                                                                  overlap_ratio=0.9)
                relevant_features = feature_vals.relevantFeatures()
                out.write('Chunk length: ' + str(chunk_length) + '\n')
                out.write('Relevant features:\n')
                out.write(str(list(relevant_features.columns)[:-2]) + '\n')

                classifier = LC.LinearClassifier(relevant_features, C_val=1, model_type='svm')
                acc, conf_matr = classifier.k_fold_train_classifier(
                    mice_treat_file_ids=feature_vals.mice_treat_file_ids, log=out)

                out.write('\n\nClassification accuracy:\n')
                out.write(str(acc) + '\n')
                out.write('Confusion matrix:\n')
                out.write(str(conf_matr) + '\n')

                out.write('\n\n\n')
            except Exception as ex:
                raise ex
