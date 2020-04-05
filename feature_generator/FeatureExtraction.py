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

    def __init__(self, feature_dict, md ,signal_type, mouse_ids={165, 166}, slice_min=30, target="treatment", part_last=10):
        self.fc_parameters = feature_dict
        self.mouse_data = md
        self.signal_type = signal_type
        self.mouse_ids = mouse_ids
        self.chunk_duration = part_last
        self.data_preparation(signal_type, slice_min, target, part_last)



    def getfewFeatures(self, mouse_id=165, signal_type='running', signal_treat='eth', slice_min=0):
        signal = self.mouse_data.fetch_mouse_signal(mouse_id, signal_treat, signal_type)
        signal = signal.sliced_data(slice_min=slice_min)
        signal = signal.get_pandas(time=True)

        id = np.repeat(1, len(signal))
        id[signal.iloc[:, 0] < 30] = 0

        #fc_parameters = MinimalFCParameters()

        signal.insert(0, "id", id, True)
        signal.columns = ["id", "time", signal_type]

        extracted_features = extract_features(signal, column_id='id', column_sort='time', column_value=signal_type, default_fc_parameters=self.fc_parameters)

        return extracted_features

    def getFeatures(self, brain_half = 'left'):
        if brain_half == 'right':
            column_value = self.mouse_data.col_names[self.signal_type][2]
        elif brain_half == 'both':
            column_value = self.mouse_data.col_names[self.signal_type][1:3]
        else:
            column_value = self.mouse_data.col_names[self.signal_type][1]
        extracted_features = extract_features(self.collected_data, column_id='id', column_sort='time_min',
                         column_value= column_value,
                         default_fc_parameters= self.fc_parameters)
        return extracted_features

    def relevantFeatures(self):
        features_filtered_direct = extract_relevant_features(self.collected_data, y = self.y, column_id='id', column_sort='time_min', default_fc_parameters=self.fc_parameters)
        features_filtered_direct["target_class"] = self.y
        return features_filtered_direct


    def data_preparation(self, signal_type='running', slice_min=30, target="treatment", part_last=10):
        mouse_ids = self.mouse_ids
        mouse_map = list(iter.product(mouse_ids, self.mouse_data.treatments))
        target_map = []
        target_y = []

        for j in mouse_map:
            data_gen = self.mouse_data.fetch_mouse_signal(j[0], j[1], signal_type)
            data = data_gen.sliced_data(slice_min=slice_min)
            chuncks = data.partition_data(part_last=part_last)

            chunck_itterator = 0
            for chunck in chuncks:
                chunck = chunck.get_pandas(time=True)

                # sometimes chuncks have length 0 and we need to skip those chuncks
                if not len(chunck):
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