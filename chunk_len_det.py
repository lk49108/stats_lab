import data_scheduler.lib_data_merger as mice_data
import feature_generator.FeatureExtraction as feature_generator
from tsfresh.feature_extraction.settings import EfficientFCParameters, ComprehensiveFCParameters
import classifier.LinearClassifier as LC

mice_data_dir = r'C:\Users\lkokot\Desktop\ETHZ_STAT_MSC\sem_2\stats_lab\analysis\CSV data files for analysis'
out_file = r'C:\Users\lkokot\Desktop\ETHZ_STAT_MSC\sem_2\stats_lab\analysis\log.out'

mice_ids = {126, 165, 166, 167, 170, 299, 302, 303, 306, 307, 323, 327}
validation_mice = {165, 170, 307, 327}
training_mice = {166, 170, 299, 306}
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
if __name__=='__main__':
    md = mice_data.MiceDataMerger(mice_data_dir)

    features = EfficientFCParameters()

    with open(out_file, 'w') as out:

        for chunk_length in range(4, 1, -1):
            try:
                print('Chunk length:', chunk_length)
                feature_vals = feature_generator.FeatureExtractor(features,
                                                                       md,
                                                                       'brain_signal',
                                                                       brain_half='right',
                                                                       mouse_ids=mice_ids,
                                                                       slice_min=30,
                                                                       target='all_vs_all',
                                                                       part_last=chunk_length,
                                                                       equal_length=True,
                                                                       slice_end=60,
                                                                       overlap_ratio=0.8)
                relevant_features = feature_vals.relevantFeatures()
                out.write('Chunk length: ' + str(chunk_length) + '\n')
                out.write('Relevant features:\n')
                out.write(str(list(relevant_features.columns)[:-1])+'\n')

                classifier = LC.LinearClassifier(relevant_features, C_val=1)
                acc, conf_matr = classifier.k_fold_train_classifier()

                out.write('Classification accuracy:\n')
                out.write(str(acc)+'\n')
                out.write('Confusion matrix:\n')
                out.write(str(conf_matr)+'\n')

                out.write('\n\n')
            except Exception as ex:
                raise ex
