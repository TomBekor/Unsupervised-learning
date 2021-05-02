import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.preprocessing import StandardScaler


def hand_postures_preprocess(path, sample, anomaly_detection):
    data = pd.read_csv(path, na_values='?').drop(0)
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

    data_anomalies = None
    target_anomalies = None
    if anomaly_detection:
        clf = IsolationForest(random_state=0).fit(data)
        predictions = clf.predict(data)
        pos_mask = predictions != -1
        neg_mask = predictions == -1
        clean_data = pd.DataFrame(data.values[pos_mask, :], columns=data.columns)
        data_anomalies = pd.DataFrame(data.values[neg_mask, :], columns=data.columns)
        clean_target = pd.DataFrame(target.values[pos_mask, :], columns=target.columns)
        target_anomalies = pd.DataFrame(target.values[neg_mask, :], columns=target.columns)
        data = clean_data
        target = clean_target


    # scaler = StandardScaler().fit(data)
    # scaled_data = scaler.transform(data)
    # data = pd.DataFrame(scaled_data, columns=data.columns)

    data = sample_df(data, sample)
    target = sample_df(target, sample)

    return data, target, data_anomalies, target_anomalies


def pulsar_stars_preprocess(path, sample, anomaly_detection):

    columns = [
        'Mean of the integrated profile',
        'Standard deviation of the integrated profile',
        'Excess kurtosis of the integrated profile',
        'Skewness of the integrated profile',
        'Mean of the DM-SNR curve',
        'Standard deviation of the DM-SNR curve',
        'Excess kurtosis of the DM-SNR curve',
        'Skewness of the DM-SNR curve',
        'Class'
    ]
    data = pd.read_csv(path, header=None)
    data.columns = columns
    target = pd.DataFrame(data.pop('Class'))

    data_anomalies = None
    target_anomalies = None
    if anomaly_detection:
        clf = IsolationForest(random_state=0).fit(data)
        predictions = clf.predict(data)
        pos_mask = predictions != -1
        neg_mask = predictions == -1
        clean_data = pd.DataFrame(data.values[pos_mask, :], columns=data.columns)
        data_anomalies = pd.DataFrame(data.values[neg_mask, :], columns=data.columns)
        clean_target = pd.DataFrame(target.values[pos_mask, :], columns=target.columns)
        target_anomalies = pd.DataFrame(target.values[neg_mask, :], columns=target.columns)
        data = clean_data
        target = clean_target

    scaler = StandardScaler().fit(data)
    scaled_data = scaler.transform(data)
    data = pd.DataFrame(scaled_data, columns=data.columns)

    data = sample_df(data, sample)
    target = sample_df(target, sample)

    return data, target, data_anomalies, target_anomalies


def sample_df(df, sample):
    return df.sample(n=sample, random_state=100)
