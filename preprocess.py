import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def hand_postures_preprocess(path, sample):

    data = pd.read_csv(path, na_values='?')
    target = pd.DataFrame(data['Class'])
    data.pop('Class')
    data.pop('User')
    drop = ['X7', 'Y7', 'Z7', 'X8', 'Y8', 'Z8', 'X9', 'Y9', 'Z9', 'X10', 'Y10', 'Z10', 'X11', 'Y11', 'Z11']
    data = data.drop(drop, axis=1)
    cols_means = {}
    describe = data.describe()
    for col in data.columns:
        cols_means[col] = describe[col]['mean']
    for col in data.columns:
        data[col] = data[col].fillna(cols_means[col])

    data = sample_df(data, sample)
    target = sample_df(target, sample)

    return data, target

# TODO second data preprocess.


def sample_df(df, sample):
    return df.sample(n=sample, random_state=100)
