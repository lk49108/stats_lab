import numpy as np
import pandas as pd
import itertools as iter
from tsfresh import extract_relevant_features, extract_features
from tsfresh.feature_selection.selection import select_features
from tsfresh.feature_extraction.settings import from_columns
import data_scheduler.lib_data_merger as lib_merger
#these are the different settings for the functions
#from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters, ComprehensiveFCParameters


class FeatureExtractor:

    fc_parameters = {}
    mouse_data = []
    y = pd.Series()
    collected_data = pd.DataFrame()

    def __init__(self, feature_dict, md, signal_type, brain_half = 'left', mouse_ids={165, 166}, slice_min=30,
                 target="treatment", part_last=10, equal_length = True, slice_end = 60, overlap_ratio = 0):
        self.fc_parameters = feature_dict
        self.mouse_data = md
        self.signal_type = signal_type
        self.mouse_ids = mouse_ids
        self.chunk_duration = part_last
        self.data_preparation(signal_type, slice_min, target, part_last, equal_length, slice_end, overlap_ratio)
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
        extracted_features['target_class'] = self.y
        return extracted_features

    def getFeatures2(self, target_class = False):
        extracted_features = extract_features(self.collected_data, column_id='id', column_sort='time_min',
                         column_value= self.column_value,
                         default_fc_parameters= self.fc_parameters)
        extracted_features.selection_type = 'all'
        if target_class:
            extracted_features['target_class'] = self.y
        X_filtered = select_features(extracted_features, self.y, ml_task = 'classification')
        kind_to_fc_parameters = from_columns(X_filtered)
        return kind_to_fc_parameters


    def relevantFeatures(self):
        features_filtered_direct = extract_relevant_features(self.collected_data, y = self.y, column_id='id', column_sort='time_min',
                                                             column_value= self.column_value, default_fc_parameters=self.fc_parameters)
        print('Identified ', len(features_filtered_direct.columns), ' relevant features.')
        features_filtered_direct['target_class'] = self.y
        features_filtered_direct.selection_type = 'relevant'
        return features_filtered_direct


    def data_preparation(self, signal_type, slice_min, target, part_last, equal_length, slice_end, overlap_ratio=0):
        if target is not 'nea_vs_all' and target is not 'all_vs_all':
            raise ValueError('The target argument must be either nea_vs_all or all_vs_all')
        if overlap_ratio is None:
            overlap_ratio = 0

        mouse_map = list(iter.product(self.mouse_ids, self.mouse_data.treatments))
        target_map = []
        target_y = []
        chuncks = []
        file_id = []
        previous_ids = set()

        for j in mouse_map:
            data_gen = self.mouse_data.fetch_mouse_signal(j[0], j[1], signal_type)
            if data_gen is None:
                continue
            if isinstance(data_gen, (lib_merger.DataFrame)):
                data_gen=[data_gen]

            file_itterator = 0
            for data in data_gen:
                data = data.right_slice(slice_min = slice_min).left_slice(slice_min = slice_end)
                chuncks = data.partition_data(part_last=part_last, overlap = True, overlap_ratio = overlap_ratio)

                chunck_itterator = 0
                file_itterator += 1
                for chunck in chuncks:
                    chunck = chunck.get_pandas(time=True)

                    # sometimes chuncks have length 0 and we need to skip those chuncks
                    if not len(chunck):
                        continue

                    chunck_itterator += 1
                    current_id = str(chunck_itterator) + '-' + str(j[0]) + '_' + str(j[1])

                    # id contains fileid-chunckid-mouseid_treatmentclass
                    id = np.repeat(str(file_itterator) + '-' + current_id, len(chunck))
                    chunck.insert(0, 'id', id, True)

                    # all stacked in rows
                    self.collected_data = pd.concat([self.collected_data, chunck], axis=0)
                    target_map.append(id[1])
                    target_y.append(str(j[1]))

        # hand to target y the class we want to predict, should not contain sample ids
        self.y = pd.Series(index=target_map, data=target_y)

        # classify treatment or no treatment
        # if false all types of treatments are considered
        if target == 'nea_vs_all':
            self.y[self.y.values != 'nea'] = 'treat'