import data_scheduler.lib_data_merger as mice_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tsfresh import extract_features

fc_parameters = {
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


def plotFeatures(feature_data):
    #feature_data = featureDataPreparation(mouse, signal_type, brain_half, mouse_ids, treatments)
    feature_list, feature_data, mouse_ids, treatments, chunk_duration = feature_data
    print(feature_data)
    feature_number = len(feature_data.columns)
    subplot_position = 1
    for j in mouse_ids:
        i = 0
        indices = feature_data.index
        length = len(indices)
        matched_indexes = []
        while i < length:
            if str(j) in indices[i]:
                matched_indexes.append(i)
            i += 1

        mouse = feature_data.iloc[matched_indexes,:]
        indices = mouse.index
        length = len(indices)
        list_indexes = []
        for k in range(len(treatments)):
            i = 0
            matched_indexes = []
            while i < length:
                if str(j)+'.'+str(k+1) in indices[i]:
                    matched_indexes.append(i)
                i += 1
            list_indexes.append(matched_indexes)
            for f in range(feature_number):
                    fig = plt.figure(num = f+1, figsize=(12,12))
                    ax = fig.add_subplot(2, 2,subplot_position)
                    ax.plot(list(range(len(matched_indexes))), mouse.iloc[matched_indexes,f], label = str(treatments[k]))
                    plt.title('Mouse '+str(j))
                    plt.xlabel("Data chunk ("+str(chunk_duration)+' Min.)')
                    plt.ylabel("Feature value")
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                    fig.suptitle(str(feature_data.columns[f]), fontsize=16, horizontalalignment='center', verticalalignment='top')
                    fig.tight_layout(pad=3.0)
        subplot_position += 1
        for ff in range(feature_number):
            fig = plt.figure(num=ff + 1, figsize=(12,12))
            axe = fig.add_subplot(2, 2, 3)
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']*len(mouse_ids)
            axe.plot(list(range(len(list_indexes[0]))), mouse.iloc[list_indexes[0],ff], list(range(len(list_indexes[1]))),
                     mouse.iloc[list_indexes[1],ff], list(range(len(list_indexes[2]))), mouse.iloc[list_indexes[2],ff],
                     list(range(len(list_indexes[3]))), mouse.iloc[list_indexes[3],ff], color = 'red')
            for i, j in enumerate(axe.lines):
                j.set_color(colors[i])
            plt.title('Mice ' + str(mouse_ids))
            plt.xlabel("Data chunk ("+str(chunk_duration)+' Min.)')
            plt.ylabel("Feature value")
            plt.legend(treatments , loc='upper right') #[a,b,c,d],treatments , loc='upper right'
    plt.show()

'''for j in mouse_ids:
        mouse_specific = feature_data.iloc[str(j) in indices,0]
        print(mouse_specific)'''
'''for k in feature_list:
            bool = str(j) in k
            list.append(bool)
            print(type(k.index))
        print(list)'''
    #print(feature_data,'\n')
    #print(mouse_ids)
'''    print(type(feature_data))
    for j in feature_data:
        print(type(j))'''
'''for i in mouse_ids:
        for j in feature_data:
            list = []
            for k in j:
                bool = str(i) in k
                list.append(bool)
            print(list)'''

'''plt.plot(feature_data.iloc[0:3,1])
    plt.plot(feature_data.iloc[3:6, 1])
    plt.title("Medikamentenkonzentration über die Zeit")
    plt.xlabel("Zeit")
    plt.ylabel("Konzentration")
    plt.show()'''

def featureDataPreparation(mouse,chunk_duration, signal_type, brain_half = 'left', mouse_ids = [165, 166, 167], treatments = ['glu', 'eth', 'sal', 'nea']):
    if brain_half == 'right':
        column_value = mouse.col_names[signal_type][2]
    else:
        column_value = mouse.col_names[signal_type][1]
    #mouse_ids = mouse_ids
    helper = []
    for j in mouse_ids:
        for i in treatments:
            signal = mouse.fetch_mouse_signal(j, i, signal_type)
            chunks = signal.partition_data(part_last=chunk_duration, remove_shorter = True)
            n = 1
            id = []
            for chunk in chunks:
                df = chunk.get_pandas(time=True)
                if n < 10:
                    chunk_id = str(j)+'.'+str(treatments.index(i)+1)+'.0'+str(n)
                else:
                    chunk_id = str(j)+'.'+str(treatments.index(i)+1)+'.'+str(n)
                n += 1
                chunk_id = np.repeat(chunk_id, len(df))
                id = np.append(id,chunk_id)
            chunk_data = signal.get_pandas()[0:len(id)]
            chunk_data.insert(0, "id", id, True)
            extracted_features = extract_features(chunk_data, column_id='id', column_sort='time_min',
                                                  column_value= column_value,
                                                  default_fc_parameters=fc_parameters)
            helper.append(extracted_features)
    feature_data = pd.concat(helper)
    return(helper, feature_data, mouse_ids, treatments, chunk_duration)



'''for i in colnames:
     helper.append(extracted_features.iloc[:,i])'''
# feature_dataframe[i] = extracted_features.iloc[:,colnames.index(i)] #helper.append(extracted_features.iloc[:,colnames.index(i)])
# print(feature_dataframe)
# feature_dataframe.append(extracted_features)
# colnames = list(extracted_features.columns)
if __name__=='__main__':
    mice_data_dir = r'C:\Users\marce\OneDrive - ETHZ\Education\ETH\Spring 2020\Statslab\Project_Neuroscience\dataset'
    md = mice_data.MiceDataMerger(mice_data_dir)
    feature_data = featureDataPreparation(md, chunk_duration=10, signal_type='brain_signal')
    plotFeatures(feature_data)
#print(len(data_eth.get_pandas()))
    '''id = np.repeat(1, len(data_eth.get_pandas()))
    id[data_eth.get_pandas().iloc[:, 0] < 30] = 0
    #print(id)
    data_eth.get_pandas().insert(0, "id", id, True)
    data_eth.get_pandas().columns = ["id", "time", 'neu_act_1', 'neu_act_2']
    extracted_features = extract_features(data_eth.get_pandas(), column_id='id', column_sort='time', column_value='neu_act_1',
                                          default_fc_parameters=fc_parameters)
    print(extracted_features)'''
'''    signal = md.fetch_mouse_signal(165, 'eth', 'brain_signal')
    signal = signal.sliced_data(29.997)
    chunks = signal.partition_data(part_last=5, remove_shorter = True)
    n = 1
    id = []
    for chunk in chunks:
        df = chunk.get_pandas(time=True)
        if n < 10:
            chunk_id = str(1)+'.'+str(1)+'.0'+str(n)
        else:
            chunk_id = str(1)+'.'+str(1) + '.' + str(n)
        n += 1
        chunk_id = np.repeat(chunk_id, len(df))
        id = np.append(id,chunk_id)
    signal = signal.get_pandas()[0:len(id)]
    signal.insert(0, "id", id, True)
    names = md.col_names['brain_signal']
    names.insert(0, 'id')
    signal.columns = names
    extracted_features = extract_features(signal, column_id='id', column_sort='time_min',
                                          column_value='neu_act_1',
                                          default_fc_parameters=fc_parameters)
    print(extracted_features)'''



'''def getfewFeatures(data, print_feat=False, signal="neu_act_1"):
    id = np.repeat(1, len(data))
    id[data.iloc[:, 0] < 30] = 0

    #fc_parameters = MinimalFCParameters()
    fc_parameters = {
        "large_standard_deviation": [{"r": 0.05}, {"r": 0.1}],
        "abs_energy": None,
        "autocorrelation": [{"lag" : 3}],
        "variance": None,
        "number_peaks": [{"n" : 10}],
        "count_above_mean": None,
        "longest_strike_below_mean": None,
        "mean": None,
        "maximum": None,
        "median": None,
        "variance" : None
    }

    data.insert(0, "id", id, True)
    data.columns = ["id", "time", signal]
    extracted_features = extract_features(data, column_id='id', column_sort='time', column_value=signal, default_fc_parameters=fc_parameters)
    if print_feat:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(extracted_features)

    return extracted_features'''
