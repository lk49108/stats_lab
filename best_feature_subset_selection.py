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

    with open(out_file, 'w') as out:

        for part_last in [15, 16, 17]:
            out.write('Chunk length: ' + str(part_last))
            all_features = ComprehensiveFCParameters()
            feature_vals = feature_generator.FeatureExtractor(all_features,
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
                                                                  part_last=part_last,
                                                                  equal_length=True,
                                                                  overlap_ratio=0.9)


            models, acc_progression = [], []
            cur_model, k = {}, 0
            while len(all_features) > 0 and k<=10:
                k+=1
                next_best_acc, next_best_feat, next_conf_matr = -1, None, None
                for feature in all_features.items():
                    cur_model[feature[0]] = feature[1]

                    try:
                        features = feature_vals.getFeatures(feature_dict=cur_model)
                        classifier = LC.LinearClassifier(features, C_val=1, model_type='svm')
                        acc, conf_matr = classifier.k_fold_train_classifier(mice_treat_file_ids=feature_vals.mice_treat_file_ids)
                    except Exception as ex:
                        print(ex)
                        exit(0)

                    if acc > next_best_acc:
                        next_best_acc = acc
                        next_best_feat = feature
                        next_conf_matr = conf_matr

                    del cur_model[feature[0]]

                cur_model[next_best_feat[0]] = next_best_feat[1]
                del all_features[next_best_feat[0]]

                acc_progression.append(next_best_acc)

                out.write('Number of features in model: ' + str(len(cur_model)) + '\n')
                out.write('Current model:\n' + str(cur_model) + '\n')
                out.write('Model accuracy: ' + str(next_best_acc) + '\n')
                out.write('Model confusion matrix:\n')
                out.write(str(next_conf_matr))
                out.write('\n\n')

            out.write('\n')
            out.write('Accuracy progression: ' + str(acc_progression))
            out.write('\n\n\n')