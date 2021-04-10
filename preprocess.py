import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def data3(path, sample):
    target = ['country']
    data = pd.read_csv(path, delimiter=";")
    data = data.sample(n=sample, random_state=100)
    target = data[target]
    data = data.drop(target, axis=1)
    data = data.drop(['year', 'page 2 (clothing model)'], axis=1)
    scaler = StandardScaler()
    data = pd.DataFrame(data=scaler.fit_transform(data), columns=[data.columns])

    return data, target


def data2(path, sample):
    def race_map(r):
        if r == 'Caucasian': return 0
        elif r == 'AfricanAmerican': return 1
        elif r == 'Asian': return 2
        elif r == 'Hispanic': return 3
        elif r == 'Other': return 4
        else: return 5

    def gender_map(g):
        if g == 'Male': return 0
        elif g == 'Male': return 1
        else: return 2

    def age_map(a):
        return int(a[a.find('-') + 1: -1]) - 5

    def weight_map(w):
        if str(w) == 'nan': return float('nan')
        elif w[0] == '>': return float(250)
        else: return float(w[w.find('-') + 1: -1])

    def str_map(s):
        return str(s)

    def weight_std_map(w):
        if str(w) == 'nan':
            return w_std
        return w

    def id1_map(id1):
        if str(id1) == 'nan':
            return id1top
        return id1

    def id2_map(id2):
        if str(id2) == 'nan':
            return id2top
        return id2

    def id3_map(id3):
        if str(id3) == 'nan':
            return id3top
        return id3

    data = pd.read_csv(path, na_values=['?', 'nan'], low_memory=False)

    # data = data[data['medical_specialty'].notnull()]

    # target preprocess:
    target = ['race', 'gender']
    target_df = data[target]

    gender = target_df['gender'].map(gender_map)
    gender_df = pd.DataFrame(gender)

    race = target_df['race'].map(race_map)
    race_df = pd.DataFrame(race)

    target_df = pd.concat([gender_df, race_df], axis=1)

    useless_features = ['encounter_id', 'patient_nbr', 'payer_code']
    low_information_features = ['medical_specialty']

    data = data.drop(target + useless_features + low_information_features, axis=1)

    # data separation to numeric and categorical values:

    age = data['age'].map(age_map)
    age_df = pd.DataFrame(age)

    weight = data['weight'].map(weight_map)
    weight_df = pd.DataFrame(weight)
    w_std = weight_df.describe()['weight']['std']
    weight_df = pd.DataFrame(weight_df['weight'].map(weight_std_map))

    data = data.drop(['age', 'weight'], axis=1)
    data = pd.concat([age_df, weight_df, data], axis=1)

    numeric_features = ['age', 'weight', 'time_in_hospital', 'num_lab_procedures',
                        'num_procedures', 'num_medications', 'number_outpatient',
                        'number_emergency', 'number_inpatient', 'number_diagnoses']

    numeric_df = data[numeric_features]
    cat_df = data.drop(numeric_features, axis=1)

    # numeric values normalization:
    scaler = StandardScaler()
    numeric_df = pd.DataFrame(data=scaler.fit_transform(numeric_df), columns=[numeric_df.columns])

    # categorical values one_hot and null completion:
    ids = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']

    id_1_df = cat_df['admission_type_id'].map(str_map)
    id_2_df = cat_df['discharge_disposition_id'].map(str_map)
    id_3_df = cat_df['admission_source_id'].map(str_map)

    diags = ['diag_1', 'diag_2', 'diag_3']

    id1top = cat_df['diag_1'].describe()['top']
    diag_1_df = cat_df['diag_1'].map(id1_map)

    id2top = cat_df['diag_2'].describe()['top']
    diag_2_df = cat_df['diag_2'].map(id2_map)

    id3top = cat_df['diag_3'].describe()['top']
    diag_3_df = cat_df['diag_3'].map(id3_map)

    cat_df = cat_df.drop(ids + diags, axis=1)
    cat_df = pd.concat([id_1_df, id_2_df, id_3_df,
                        diag_1_df, diag_2_df, diag_3_df,
                        cat_df], axis=1)

    one_hot_df = pd.get_dummies(cat_df)

    data = pd.concat([numeric_df, one_hot_df], axis=1)

    data = data.sample(n=sample, random_state=100)
    target_df = target_df.sample(n=sample, random_state=100)

    return data, target_df
