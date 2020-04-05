import matplotlib.pyplot as plt
from FeatureExtraction import getfewFeatures
import numpy as np
import pandas as pd
import seaborn as sns

def plotFeatures(mouse, signal='running'):

    mouse_ids = [165, 166]

    general_features = pd.DataFrame()
    for i in mouse.treatments:
        for j in mouse_ids:
            data_gen = mouse.fetch_mouse_signal(j, i, signal)
            data = data_gen.sliced_data(time=True, slice_min=30)
            features = getfewFeatures(data, signal=signal)
            features["treatment"] = i
            features["mouse_id"] = j
            general_features = pd.concat([general_features, features], axis=0)

    if len(features) == 0:
        raise ValueError('No features extracted.')

    print(general_features.columns)

    sns.set(style="ticks", color_codes=True)
    sns.catplot(x="treatment", y=signal + '__sum_values', data=general_features)
    plt.show()