import numpy as np
import pandas as pd
import itertools as iter
from tsfresh import extract_relevant_features, extract_features
#these are the different settings for the functions
#from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters, ComprehensiveFCParameters


class FeatureExtractor:

    fc_parameters = {}
    mouse_data = []
    y = pd.Series()
    collected_data = pd.DataFrame()

    def __init__(self, feature_dict, md ,signal_type, brain_half = 'left', mouse_ids={165, 166}, slice_min=30,
                 target="treatment", part_last=10, equal_length = True):
        self.fc_parameters = feature_dict
        self.mouse_data = md
        self.signal_type = signal_type
        self.mouse_ids = mouse_ids
        self.chunk_duration = part_last
        self.data_preparation(signal_type, mouse_ids, slice_min, target, part_last, equal_length)
        if signal_type == 'brain_signal' and brain_half == 'right':
            column_value = self.mouse_data.col_names[self.signal_type][2]
        elif signal_type == 'brain_signal' and brain_half == 'both':
            column_value = self.mouse_data.col_names[self.signal_type][1:3]
        else:
            column_value = self.mouse_data.col_names[self.signal_type][1]
        self.column_value = column_value


    def getFeatures(self, target_class = False):
        extracted_features = extract_features(self.collected_data, column_id='id', column_sort='time_min',
                         column_value= self.column_value,
                         default_fc_parameters= self.fc_parameters)
        extracted_features.selection_type = 'all'
        if target_class:
            extracted_features['target_class'] = self.y
        return extracted_features

    def relevantFeatures(self):
        features_filtered_direct = extract_relevant_features(self.collected_data, y = self.y, column_id='id', column_sort='time_min',
                                                             column_value= self.column_value, default_fc_parameters=self.fc_parameters)
        features_filtered_direct["target_class"] = self.y
        features_filtered_direct.selection_type = 'relevant'
        return features_filtered_direct


    def data_preparation(self, signal_type, mouse_ids, slice_min, target, part_last, equal_length):
        mouse_map = list(iter.product(mouse_ids, self.mouse_data.treatments))
        target_map = []
        target_y = []
        chuncks = []
        length_subtractor = 0

        for j in mouse_map:
            data_gen = self.mouse_data.fetch_mouse_signal(j[0], j[1], signal_type)
            data = data_gen.sliced_data(slice_min=slice_min)
            previous_chunk_length = len(chuncks)
            chuncks = data.partition_data(part_last=part_last)
            if equal_length:
                if j[1] == 'nea':
                    length = previous_chunk_length - length_subtractor
                else:
                    length = len(chuncks)
            else:
                length = len(chuncks)+1

            chunck_itterator = 0
            for chunck in chuncks[0:length-1]:
                chunck = chunck.get_pandas(time=True)

                # sometimes chuncks have length 0 and we need to skip those chuncks
                if not len(chunck):
                    length_subtractor = 1
                    continue

                chunck_itterator = chunck_itterator + 1
                # id contains chunckid-mouseid_treatmentclass
                id = np.repeat(str(chunck_itterator) + '-' + str(j[0]) + '_' + str(j[1]), len(chunck))
                chunck.insert(0, 'id', id, True)

                # all stacked in rows
                self.collected_data = pd.concat([self.collected_data, chunck], axis=0)
                target_map.append(id[1])
                target_y.append(str(j[1]))

        # hand to target y the class we want to predict, should not contain sample ids
        self.y = pd.Series(index=target_map, data=target_y)
        # classify treatment or no treatment
        # if false all types of treatments are considered
        if target == 'treatment':
            self.y[self.y.values != 'nea'] = 'treat'