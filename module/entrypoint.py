from module.data import TrainDataset, train_transform, val_transform, get_tab_from_patient_data, \
    generate_crops_from_lead
from module.model import ResNet
from module.dataframe import get_df_from_patients, add_folds
from helper_code import find_patient_files, get_locations
import numpy as np
from torch.utils.data import DataLoader, Dataset
from module.engines import Fitter
import glob
import torch
from tqdm.auto import tqdm
from module.threshold_search import generate_matrix
from module.metamodel import train_metamodel
import pandas as pd

from module.config import TrainGlobalConfig

config = TrainGlobalConfig()


def train(data_folder, model_folder, verbose, continue_training=False):
    config.folder = model_folder
    df = get_df_from_patients(find_patient_files(data_folder))
    if len(df) < 750:
        raise ValueError('Problem with train-data loading.')
    df = add_folds(df, num_folds=5, random_state=0)

    allowed_folds = np.arange(5)
    for val_f, thr_f in zip(allowed_folds, np.roll(allowed_folds, -1)):
        print("Val fold:", val_f, "threshold fold:", thr_f)

        train_patients = []
        train_patients.extend(df[~df.fold.isin([val_f, thr_f])]["file"].values)
        train_patients.extend(df[(~df.fold.isin([val_f, thr_f])) & (df["murmur"] == "Unknown")]["file"].values)
        train_patients.extend(df[(~df.fold.isin([val_f, thr_f])) & (df["murmur"] == "Unknown")]["file"].values)

        val_patients = df[df.fold == val_f]["file"].values
        threshold_selection_patients = df[df.fold == thr_f]["file"].values

        assert (pd.Series(val_patients).isin(threshold_selection_patients)).sum() == 0
        assert (pd.Series(val_patients).isin(train_patients)).sum() == 0
        assert (pd.Series(threshold_selection_patients).isin(train_patients)).sum() == 0

        ds_train = TrainDataset(train_patients, data_folder, transform=train_transform, mode="train")
        ds_val = TrainDataset(val_patients, data_folder, transform=val_transform, mode="val")
        ds_thresh = TrainDataset(threshold_selection_patients, data_folder, transform=val_transform, mode="val")

        train_loader = DataLoader(ds_train, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)
        val_loader = DataLoader(ds_val, batch_size=1, num_workers=config.num_workers, shuffle=False)
        thresh_loader = DataLoader(ds_thresh, batch_size=1, num_workers=config.num_workers, shuffle=False)

        model1 = ResNet(use_tab=False).to(config.device)
        fitter1 = Fitter(model1, config.device, config)

        for j, fitter in enumerate([fitter1]):
            print("training model", j)

            if continue_training:
                best_cp_kind = "auc_murmur"
                models_paths = glob.glob(config.folder + f"/best-{best_cp_kind}-model{j}_fold0_{val_f}-*epoch.bin")
                assert len(models_paths) == 1
                models_path = models_paths[0]
                fitter.load(models_path)

            result = fitter.fit(train_loader, val_loader, model_nr=j, fold="0", innerfold=val_f)

            best_cp_kind = "auc_murmur"
            models_paths = glob.glob(config.folder + f"/best-{best_cp_kind}-model{j}_fold0_{val_f}-*epoch.bin")
            assert len(models_paths) == 1
            models_path = models_paths[0]
            fitter.load(models_path)

            metamodel = False
            if not metamodel:
                print("Finding best threshold")
                threshold_list, matrix = generate_matrix(thresh_loader, fitter, val_f, j)
                fitter.best_thresholds_murmur = threshold_list
                fitter.save(models_path)
            else:
                train_metamodel(df, fitter, threshold_selection_patients)


def load_model(model_folder, verbose):
    config = TrainGlobalConfig()
    config.folder = model_folder
    model1 = ResNet(use_tab=False).to(config.device)
    fitter1 = Fitter(model1, config.device, config)
    return [fitter1], model_folder


label_counter = [0, 0, 0]
target_dist = [0.19, 0.07, 0.74]


def run_model(model, data, recordings, verbose):
    fitters_list, model_folder = model
    del model

    allprobas_murmur = []
    allprobas_outcome = []
    best_threshs_unknown = {"all": []}
    best_threshs_absent = {"all": []}
    threshold_outcome = []

    hardvotes = []
    for j, fitter in enumerate(fitters_list):
        print("training model", j)

        models_paths = glob.glob(model_folder + f"/best-auc_murmur-model{j}_fold0_*epoch.bin")
        if len(models_paths) == 0:
            raise ValueError('No model was found.')
        print("Found", len(models_paths), "models")

        for path in models_paths:
            fitter.load(path, silent=True)
            thresholds = fitter.best_thresholds_murmur

            best_threshs_unknown["all"].append(thresholds[0][0])
            best_threshs_absent["all"].append(thresholds[0][1])
            threshold_outcome.append(thresholds[1][0])

            model = fitter.model

            probas_murmur, probas_outcome = predict_probas(fitter.config, data, model, recordings)
            allprobas_murmur.append(probas_murmur)
            allprobas_outcome.append(probas_outcome)

            label_murmur = [1, 0, 0]
            if probas_murmur[2] > thresholds[0][1]: label_murmur = [0, 0, 1]
            if probas_murmur[1] > thresholds[0][0]: label_murmur = [0, 1, 0]
            hardvotes.append(np.argmax(label_murmur))

    meaned_probas_murmur = np.array(allprobas_murmur).mean(axis=0)
    meaned_probas_outcome = np.array(allprobas_outcome).mean(axis=0)
    probabilities = np.concatenate((meaned_probas_murmur, meaned_probas_outcome))

    threshold_absent = np.mean(best_threshs_absent["all"]).round(2)
    threshold_unknown = np.mean(best_threshs_unknown["all"]).round(2)
    threshold_outcome = np.mean(threshold_outcome).round(2)

    if np.sum(label_counter) > 25:
        if percentages[2] < 0.7:  # absent
            threshold_absent -= 0.05
        if percentages[0] > 0.26:
            threshold_unknown -= 0.05

    hardvoting = False
    if not hardvoting:
        label_murmur = [1, 0, 0]
        if meaned_probas_murmur[2] > threshold_absent: label_murmur = [0, 0, 1]
        if meaned_probas_murmur[1] > threshold_unknown: label_murmur = [0, 1, 0]
    else:
        def most_common(lst):
            return max(set(lst), key=lst.count)

        label_murmur = np.zeros(3)
        print(most_common(hardvotes))
        label_murmur[most_common(hardvotes)] = 1

    label_counter[np.argmax(label_murmur)] += 1
    percentages = np.array(np.array(label_counter) / np.sum(label_counter)).round(2)
    print(threshold_absent, threshold_unknown, label_murmur, percentages)

    if meaned_probas_outcome[0] > threshold_outcome:
        label_outcome = [1, 0]
    else:
        label_outcome = [0, 1]

    labels = np.concatenate((label_murmur, label_outcome))

    return ['Present', 'Unknown', 'Absent', 'Abnormal', 'Normal'], labels, probabilities


def predict_probas(config, data, model, recordings):
    current_recordings = recordings
    current_patient_data = data
    valves = np.array(get_locations(current_patient_data))

    img_list = []
    for l in ['MV', 'AV', 'PV', 'TV']:
        picks = np.where(valves == l)[0]

        if len(picks) > 0:
            pick = picks[0]
            img = generate_crops_from_lead(current_recordings, val_transform, "val", pick)
        else:
            n_crops = config.n_crops_in_val
            img = np.zeros((n_crops, config.cropy, config.cropx, 3))
        img_list.append(img)

    img_list = torch.tensor(np.array(img_list)).unsqueeze(0).float().to(config.device)
    tab = torch.tensor(get_tab_from_patient_data(data)).unsqueeze(0).float().to(config.device)
    # tab = torch.tensor([0,0,0,0]).float().to(config.device)
    model.eval()

    TTA = False

    if TTA == True:
        img_list = torch.concat([img_list for i in range(5)])
        for i in range(4):
            img_list[i][i] = torch.zeros_like(img_list[i][i])

    with torch.no_grad():
        out = model(img_list, tab)
        probas_murmur = torch.mean(out[0], dim=0)
        probas_outcome = torch.mean(out[1], dim=0)

    avg_probas_murmur = probas_murmur.softmax(axis=0).detach().cpu().numpy()
    avg_probas_outcome = probas_outcome.softmax(axis=0).detach().cpu().numpy()
    return avg_probas_murmur, avg_probas_outcome
