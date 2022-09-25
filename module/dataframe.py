
import pandas as pd
from helper_code import *
import numpy as np
import os
from tqdm.auto import tqdm
import helper_code
from sklearn.model_selection import StratifiedKFold
import glob
import shutil


from evaluate_model import find_challenge_files, load_murmurs, load_classifier_outputs


def get_distolic_murmur_quality(data): # 4
    for i, l in enumerate(data.split('\n')):
        if l.split(" ")[0] == "#Diastolic" and l.split(" ")[1] == "murmur" and l.split(" ")[2] == "quality:":

            #template = np.zeros(len(to_one_hot))
            #template[to_one_hot.index(l.split(" ")[-1])] = 1

            return l.split(" ")[-1]


def get_campain(data): # 4
    for i, l in enumerate(data.split('\n')):
        if l.split(" ")[0] == "#Campaign:":

            #template = np.zeros(len(to_one_hot))
            #template[to_one_hot.index(l.split(" ")[-1])] = 1

            return l.split(" ")[-1]


def get_df_from_patients(patient_files):
    tab_features = []
    for file in patient_files:
        current_patient_data = load_patient_data(file)

        oneof_absent_present_unknown = helper_code.get_murmur(current_patient_data)
        oneof_abnormal_normal = helper_code.get_outcome(current_patient_data)
        sex = helper_code.get_sex(current_patient_data)
        pregnant = helper_code.get_pregnancy_status(current_patient_data)
        weight = helper_code.get_weight(current_patient_data)
        height = helper_code.get_height(current_patient_data)
        age = helper_code.get_age(current_patient_data)
        frequency = get_frequency(current_patient_data)

        murmurquality = get_distolic_murmur_quality(current_patient_data)
        campaign = get_campain(current_patient_data)
        #if murmurquality != "nan":
        #    print("detected diastolic")
        #    continue

        if compare_strings(age, 'Neonate'):
            age = 0.5
        elif compare_strings(age, 'Infant'):
            age = 6
        elif compare_strings(age, 'Child'):
            age = 6 * 12
        elif compare_strings(age, 'Adolescent'):
            age = 15 * 12
        elif compare_strings(age, 'Young Adult'):
            age = 20 * 12
        else:
            age = float('nan')

        tab_features.append([file, oneof_absent_present_unknown, oneof_abnormal_normal,
                             sex, pregnant, weight, height, age, frequency, murmurquality, campaign])

    df = pd.DataFrame(tab_features, columns=["file", "murmur", "outcome", "sex", "pregnant", "weight",
                                             "height", "age", "frequency", "murmurquality", "campaign"])
    df.loc[df["sex"] == "Female", "sex"] = 0
    df.loc[df["sex"] == "Male", "sex"] = 1
    df["pregnant"] = df["pregnant"].astype(int)
    df["stratify_column"] = df["outcome"].apply(lambda x: -1 if x == "Abnormal" else 1)
    df["stratify_column"] = df["stratify_column"] * (df["murmur"].factorize()[0] + 1)

    # todo: impute missing values
    return df


def add_folds(df, num_folds, random_state):
    df["fold"] = np.nan
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    #arr = []
    for f, (train_id, test_id) in enumerate(skf.split(df, df["stratify_column"])):
        #arr.append([*df.iloc[train_id]["outcome"].value_counts().values,
        #            *df.iloc[train_id]["murmur"].value_counts().values])
        df.loc[test_id, "fold"] = f
    df["fold"] = df["fold"].astype(int)
    return df


def place_data(_df, dataset_path, train_folder, test_folder, test_fold):
    _df["patient_nrs"] = _df["file"].str.split("/").str[-1].str[:-4]
    train, test = _df[_df.fold != test_fold]["patient_nrs"].values, _df[_df.fold == test_fold]["patient_nrs"].values


    for f in glob.glob(train_folder + "*"):
        os.remove(f)

    for f in glob.glob(test_folder + "*"):
        os.remove(f)

    for i, t in tqdm(enumerate(train), total=len(train)):

        files = glob.glob(dataset_path + "/" + str(t) + "*")
        for f in files:
            f = f.replace("\\", "/")
            shutil.copyfile(f, train_folder + f.split("/")[-1])

    for i, t in tqdm(enumerate(test), total=len(test)):
        files = glob.glob(dataset_path + "/" + str(t) + "*")
        for f in files:
            f = f.replace("\\", "/")
            shutil.copyfile(f, test_folder + f.split("/")[-1])



def gib_mir_testlabels():
    output_folder = "./test_outputs"
    label_folder = "./test_data"
    murmur_classes = ['Present', 'Unknown', 'Absent']
    label_files, output_files = find_challenge_files(label_folder, output_folder)
    murmur_labels = load_murmurs(label_files, murmur_classes)
    return murmur_labels