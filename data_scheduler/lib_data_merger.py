import os
import pandas as pd
import re
import numpy as np

class DataFrame:

    def __init__(self, df, signal_type, chunks = None):
        if df is None:
            raise ValueError('Provided None as data frame is illegal')
        if signal_type is None or signal_type not in MiceDataMerger.signals:
            raise ValueError('{0} was provided as signal type which is illegal type of signal'.format('signal_type'))
        if chunks is None:
            chunks = [(df['time_min'].min(), df['time_min'].max())]

        self.df = df
        self.freq = MiceDataMerger.data_freq[signal_type]
        self.chunks = chunks
        self.signal_type = signal_type

    def get_pandas(self, time=True):
        if time is None:
            raise ValueError('time boolean is not allowed to have None value')

        data = self.df
        if time and 'time_min' not in list(self.df):
            raise ValueError('Can not include time_min column because there is no such column in data frame')
        elif not time and 'time_min' in list(self.df):
            col_names = list(data.columns)
            col_names.remove('time_min')
            data = data[col_names]

        return data

    def sliced_data(self, slice_min=0):
        if slice_min is None or slice_min < 0:
            raise ValueError('{0} as a minute for data to be sliced at is illegal argument'.format(slice_min))

        data = self.df[self.df["time_min"] >= slice_min]
        #nea has a default length of 153 mins, cutting nea signal to have same length as other signals  
        data = data[data["time_min"] <= 110]
        new_chunks = []
        for chunk in self.chunks:
            if chunk[1] >= slice_min:
                new_chunks.append((max(chunk[0], slice_min), chunk[1]))

        return DataFrame(data, self.signal_type, chunks=new_chunks)

    def right_slice(self, slice_min):
        if slice_min is None:
            raise ValueError('{0} as a minute for data to be sliced at is illegal argument'.format(slice_min))

        data = self.df[self.df["time_min"] >= slice_min]
        new_chunks = []
        for chunk in self.chunks:
            if chunk[1] >= slice_min:
                new_chunks.append((max(chunk[0], slice_min), chunk[1]))

        return DataFrame(data, self.signal_type, chunks=new_chunks)

    def left_slice(self, slice_min):
        if slice_min is None:
            raise ValueError('{0} as a minute for data to be sliced at is illegal argument'.format(slice_min))

        data = self.df[self.df["time_min"] <= slice_min]
        new_chunks = []
        for chunk in self.chunks:
            if chunk[0] <= slice_min:
                new_chunks.append((chunk[0], min(chunk[1], slice_min)))

        return DataFrame(data, self.signal_type, chunks=new_chunks)

    def leave_out_chunk(self, time_tuple):
        if time_tuple is None:
            raise ValueError('Leave out chunk is not supposed to be None value')

        start_time, end_time = time_tuple
        if start_time is None or end_time is None:
            raise ValueError('None values provided as time stamps')

        if start_time > end_time:
            start_time, end_time = end_time, start_time

        data = self.df
        data = data[(data['time_min'] <= start_time) | (data['time_min'] >= end_time)]

        new_chunks = []
        for chunk in self.chunks:
            s, e = chunk
            if s >= start_time and e <= end_time:
                continue
            elif start_time > s and end_time < e:
                s1, e1 = s, start_time
                s2, e2 = end_time, e
                new_chunks.append((s1, e1))
                new_chunks.append((s2, e2))
                continue
            elif start_time > s:
                e = min(start_time, e)
            elif end_time < e:
                s = max(s, end_time)

            new_chunks.append((s, e))

        return DataFrame(data, self.signal_type, chunks=new_chunks)

    def leave_out_chunks(self, leave_out):
        if leave_out is None or len(leave_out) <=0:
            raise ValueError('Provided invalid array of leave out chunks')

        df = self
        for chunk in leave_out:
              df = df.leave_out_chunk(chunk)

        return df

    def partition_data(self, part_last, overlap=True, overlap_ratio=0.2, remove_shorter = False):
        if part_last is None or part_last <=0:
            raise ValueError('{0} as partition lasting time is not legal value'.format(part_last))
        if remove_shorter is None:
            raise ValueError('remove_shorter argument is not allowed to be of value None')
        if 'time_min' not in list(self.df):
            raise ValueError('Can not partition data frame by time because there is no time column in it')
        if overlap is None:
            raise ValueError('Can not partition data frame by time because there is no time column in it')
        if overlap and (overlap_ratio is None or overlap >= 1):
            raise ValueError('Overlapping is used and therefore "overlap_ratio" argument has to be provided and be less than 1')

        diff = part_last * (1 - overlap_ratio) if overlap else part_last
        chunked_data = []
        for chunk in self.chunks:
            start, end = chunk

            s = start
            while s+part_last <= end:
                chunked_data.append(DataFrame(self.df[(self.df['time_min'] >= s) & (self.df['time_min'] < s+part_last)], self.signal_type, [(s, s+part_last)]))
                s += diff

            if not remove_shorter:
                chunked_data.append(DataFrame(self.df[(self.df['time_min'] >= s) & (self.df['time_min'] < end)], self.signal_type, [(s, end)]))

        return chunked_data

    def __str__(self):
        return self.signal_type

    def __repr__(self):
        s = self.signal_type + ' '
        for chunk in self.chunks:
            s += str(chunk) + ', '
        return s[:-2]

class DataMerger:

    def __init__(self, dir):
        if dir is None:
            raise ValueError('Directory path can not be None')
        if not os.path.isdir(dir):
            raise ValueError('Inexistent directory provided')

        self.dir = dir

    def merge_data(self):
        pass

    def get_data(self, **kwargs):
        pass


class MiceDataMerger(DataMerger):

    treatments = set(['glu', 'eth', 'sal', 'nea'])
    signals = set(['brain_signal', 'running', 'v_o2', 'v_co2', 'rq', 'heat'])

    data_freq = {
        'brain_signal': 10,
        'running': 10,
        'v_o2': 1,
        'v_co2': 1,
        'rq': 1,
        'heat': 1
    }

    col_names = {
        'brain_signal': ['time_min', 'neu_act_1', 'neu_act_2'],
        'running': ['time_min', 'run_cm_s'],
        'v_o2': ['time_min', 'lit_min'],
        'v_co2': ['time_min', 'lit_min'],
        'rq': ['time_min', 'exc'],
        'heat': ['time_min', 'cal_min']
    }

    file_regex = r'([0-9]+)-(glu|eth|nea|sal)-ig-[0-9]+_(brain_signal|heat|rq|running|v_vo2|v_o2).csv'

    def __init__(self, dir):
        super().__init__(dir)
        self.mouse_data_file_map = {}
        self.preprocess_dir()

    def preprocess_dir(self):
        for file in os.listdir(self.dir):
            file_lower=file.lower()
            res = re.match(MiceDataMerger.file_regex, file_lower)
            if res is None or res.group(1) is None:
                continue

            try:
                mouse_data_id = (int(res.group(1)), res.group(2), res.group(3))
            except ValueError | AttributeError as err:
                raise ValueError('File {0} is of wrong name format'.format(file))

            if mouse_data_id[0] is None or mouse_data_id[1] is None or mouse_data_id[2] is None\
                    or mouse_data_id[1] not in MiceDataMerger.treatments or mouse_data_id[2] not in MiceDataMerger.signals:
                raise ValueError('File {0} is of wrong name format'.format(file))

            if mouse_data_id in self.mouse_data_file_map:
                self.mouse_data_file_map[mouse_data_id].append(os.path.join(self.dir, file))
            else:
                self.mouse_data_file_map[mouse_data_id] = [(os.path.join(self.dir, file))]

    def merge_data(self):
        pass

    def fetch_mouse_signal(self, mouse_id, treat, signal):
        if mouse_id is None or treat is None or treat.lower() not in MiceDataMerger.treatments \
                or signal is None or signal.lower() not in MiceDataMerger.signals:
            return None

        treat, signal = treat.lower(), signal.lower()
        mouse_signal_file_id = (mouse_id, treat, signal)
        if mouse_signal_file_id not in self.mouse_data_file_map:
            return None

        data = [DataFrame(pd.read_csv(file, names=MiceDataMerger.col_names[signal]), signal) for file in self.mouse_data_file_map[mouse_signal_file_id]]
        if len(data)<=1:
            return data[0]

        return data

    def fetch_mouse_data(self, mice_id, treatments = treatments, signals = signals):

        def check_array_or_single_value(val, parameter_name, value_constraints = None):
            if val is None:
                raise ValueError('{0} can not be None value'.format(parameter_name))
            elif isinstance(val, (list, set, np.ndarray)) and len(val) <= 0:
                raise ValueError('{0} array can not be empty'.format(parameter_name))

            val = val if isinstance(val, (list, set, np.ndarray)) else [val]
            if value_constraints:
                for v in val:
                    if v not in value_constraints:
                        raise ValueError('{0} is not legal value for {1}'.format(v, parameter_name))

            return val

        mice_id = check_array_or_single_value(mice_id, 'Mouse ID')
        treatments = check_array_or_single_value(treatments, 'Mouse treatment', MiceDataMerger.treatments)
        signals = check_array_or_single_value(signals, 'Mouse data signal', MiceDataMerger.signals)

        data = {}
        for mice in mice_id:
            data[mice]={}
            for treatment in treatments:
                data[mice][treatment] = {}
                for signal in signals:
                    try:
                        dt = self.fetch_mouse_signal(mice, treatment.lower(), signal.lower())
                        if dt:
                            data[mice][treatment][signal] = dt
                    except ValueError:
                        pass

        for mice in mice_id:
            for treatment in treatments:
                if len(list(data[mice][treatment])) == 1:
                    data[mice][treatment] = data[mice][treatment][list(data[mice][treatment])[0]]
                elif len(list(data[mice][treatment])) == 0:
                    data[mice].pop(treatment, None)

            if len(list(data[mice])) == 1:
                data[mice] = data[mice][list(data[mice])[0]]
            elif len(list(data[mice])) == 0:
                data.pop(mice, self)

        if len(list(data)) == 1:
            data = data[list(data)[0]]
        elif len(data) == 0:
            raise ValueError('There was no data succesfully loaded')

        return data

    def get_data(self, **kwargs):
        """
        :param kwargs:mouse_id,treat,list of signals,split_num,treatment/whole_data/non_treatment
        :return:
        """
        if 'mouse_id' not in kwargs:
            raise ValueError('Mouse ID not provided')
        if 'treat' not in kwargs:
            raise ValueError('Treatment type not provided')
        if 'signals' not in kwargs or kwargs['signals'] is None or len(kwargs['signals']) <= 0:
            raise ValueError('Signal list not provided')

        data = self.fetch_mouse_data(mouse_id=kwargs['mouse_id'],
                                     treat=kwargs['treat'],
                                     signals=kwargs['signals'])
        return data
