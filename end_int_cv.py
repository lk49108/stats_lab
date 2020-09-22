import data_scheduler.lib_data_merger as mice_data
import feature_generator.FeatureExtraction as feature_generator
from tsfresh.feature_extraction.settings import EfficientFCParameters, ComprehensiveFCParameters
import classifier.LinearClassifier as LC
from itertools import product

chunk_length = 15
out_file = r'C:\Users\lkokot\Desktop\ETHZ_STAT_MSC\sem_2\stats_lab\analysis\log.out'
mice_data_dir = r'C:\Users\lkokot\Desktop\ETHZ_STAT_MSC\sem_2\stats_lab\analysis\CSV data files for analysis'
mice_ids = {126, 165, 166, 167, 170, 299, 302, 303, 306, 307, 323, 327}
model = {
    "spkt_welch_density": [{"coeff": coeff} for coeff in [2, 5, 8]],
    "fft_coefficient": [{"coeff": k, "attr": a} for a, k in
                        product(["real", "imag", "abs", "angle"], range(100))],
    'kurtosis': None,
    'binned_entropy': [{'max_bins': 10}],
    'longest_strike_above_mean': None,
    'standard_deviation': None,
    'fft_aggregated': [{'aggtype': 'centroid'}, {'aggtype': 'variance'}, {'aggtype': 'skew'}, {'aggtype': 'kurtosis'}],
    'longest_strike_below_mean': None,
    'agg_autocorrelation': [{'f_agg': 'mean', 'maxlag': 40}, {'f_agg': 'median', 'maxlag': 40}, {'f_agg': 'var', 'maxlag': 40}]
}

treat_int_end = [55, 60, 65, 70, 75, 80, 85, 90]

if __name__ == '__main__':
    with open(out_file, 'w') as out:
        md = mice_data.MiceDataMerger(mice_data_dir)
        all_features = EfficientFCParameters()

        best_model_lengths, best_model_acc = None, -1
        for eth_end in treat_int_end:
            for glu_end in treat_int_end:
                for nea_end in treat_int_end:

                    feature_vals = feature_generator.FeatureExtractor(all_features,
                                                                      md,
                                                                      'brain_signal',
                                                                      brain_half='right',
                                                                      mouse_ids=mice_ids,
                                                                      slices={
                                                                          'eth': (35, eth_end),
                                                                          'glu': (35, glu_end),
                                                                          'nea': (35, nea_end),
                                                                          'sal': (35, min(eth_end, glu_end, nea_end, 55))
                                                                      },
                                                                      target='all_vs_all',
                                                                      part_last=chunk_length,
                                                                      equal_length=True,
                                                                      overlap_ratio=0.9)
                    features = feature_vals.getFeatures(feature_dict=model)
                    classifier = LC.LinearClassifier(features, C_val=1, model_type='svm')
                    acc, conf_matr = classifier.k_fold_train_classifier(mice_treat_file_ids=feature_vals.mice_treat_file_ids, log=out)

                    out_str = 'eth_end: {}, glu_end: {}, nea_end: {}, sal_end: {}'.format(eth_end, glu_end, nea_end, min(eth_end, glu_end, nea_end, 55))
                    class_acc_str = 'Accuracy: {}'.format(acc)
                    conf_matr_str = 'Confusion matrix:\n {}'.format(conf_matr)

                    print(out_str)
                    print(class_acc_str)
                    print(conf_matr_str)

                    out.write(out_str + '\n')
                    out.write(class_acc_str + '\n')
                    out.write(conf_matr_str + '\n\n\n')

                    if best_model_lengths is None or acc > best_model_acc:
                        best_model_acc = acc
                        best_model_lengths = {'eth' : eth_end, 'glu' : glu_end, 'nea' : nea_end, 'sal' : min(eth_end, glu_end, nea_end, 55)}


        fin_res_str = 'Best model interval ends: {}\nBest model accuracy: {}'.format(best_model_lengths, best_model_acc)
        print(fin_res_str)
        out.write(fin_res_str+'\n')