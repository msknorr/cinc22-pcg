import pandas as pd
from helper_code import *
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
import torch
import helper_code
import random
from module.my_helper_code import *
from module.spectogram import get_melspec_image
import albumentations as A
from module.config import TrainGlobalConfig
import librosa
import matplotlib.pyplot as plt
import cv2

config = TrainGlobalConfig()

CROP_X = config.cropx  # time
CROP_Y = config.cropy  # n mels
N_CROPS_IN_TRAIN = config.n_crops_in_train  # pro lead
N_CROPS_IN_VAL = config.n_crops_in_val

train_transform = A.Compose([
    A.Resize(CROP_X, CROP_Y),
    A.CoarseDropout(max_holes=16, max_height=24, max_width=24, p=0.5),
],
)

val_transform = A.Compose([
    A.Resize(CROP_X, CROP_Y),
],
)


def get_melspec_new_normalization(audio):
    melspec = librosa.feature.melspectrogram(y=audio.astype(float), sr=4000, n_fft=244, hop_length=64)
    melspec = librosa.amplitude_to_db(melspec, ref=np.max)
    mellog = np.abs(melspec)  # np.log(np.abs(melspec) + 1e-9)
    melspec = librosa.util.normalize(melspec)
    melspec = 1 - np.abs(melspec)
    return melspec


def generate_crops_from_lead(current_recordings, transform=None, mode="val", idx=None):
    img = current_recordings[idx]
    img = get_melspec_image(img)

    downscalefactor = img.shape[1] / CROP_X
    small_image = cv2.resize(img.copy(), (CROP_X, int(img.shape[0] // downscalefactor)))

    to_pad = (img.shape[1] % CROP_X)
    template = np.zeros((img.shape[0], img.shape[1] + to_pad, img.shape[2]))  # pad last small crop
    template[0:img.shape[0], 0:img.shape[1], :] = img

    n_crops = template.shape[1] // CROP_X

    img_crops = []

    for i in range(n_crops):
        if (mode == "train") or (mode == "threshold"):
            rand_x = np.random.randint(0, CROP_X) if i < (n_crops - 1) else 0
        else:
            rand_x = 0
        img = template[:, rand_x + (i * CROP_X): rand_x + ((i + 1) * CROP_X), :].astype(np.uint8)

        # augmentations
        if transform is not None:
            augmented = transform(image=img)
            img = augmented['image']

        if (mode == "train") or (mode == "threshold"):
            # drop frequency
            for jj in range(2):
                if (np.random.randint(100) > 90):
                    c = np.random.randint(0, CROP_Y)
                    img[c:c + np.random.randint(5, 10), :, :] = 0
            # drop time
            for jj in range(2):
                if (np.random.randint(100) > 90):
                    c = np.random.randint(0, CROP_X)
                    img[:, c:c + np.random.randint(5, 10), :] = 0
        img[-small_image.shape[0]:, 0:small_image.shape[1], :] = small_image  # embed whole sequence in crop

        img_crops.append(img)

    img_crops = np.array(img_crops) / 255

    if mode == "train":
        np.random.shuffle(img_crops)

    pad_crop_till = N_CROPS_IN_TRAIN if mode == "train" else N_CROPS_IN_VAL
    if len(img_crops) < pad_crop_till:
        tt = np.zeros((pad_crop_till, img_crops.shape[1], img_crops.shape[2], 3))
        tt[0:len(img_crops), :] = img_crops
        return tt

    return img_crops[0:pad_crop_till]


def get_tab_from_patient_data(current_patient_data):
    ##############################################
    ###########ALERT!!!!! NOT ON SUBMISSION#######
    ##############################################

    sex = helper_code.get_sex(current_patient_data)
    if sex == "Female":
        sex = 0
    elif sex == "Male":
        sex = 1
    pregnant = int(helper_code.get_pregnancy_status(current_patient_data))
    weight = helper_code.get_weight(current_patient_data)
    height = helper_code.get_height(current_patient_data)
    age = helper_code.get_age(current_patient_data)

    # sex
    sex1hot = np.zeros(2, dtype=float)
    sex1hot[sex] = 1

    # pregnant
    pregnant1hot = np.zeros(2, dtype=float)
    pregnant1hot[pregnant] = 1

    # age
    if age == "nan":
        age = "Child"
    if compare_strings(age, 'Neonate'):
        age = [1, 0, 0, 0, 0]
    elif compare_strings(age, 'Infant'):
        age = [1, 1, 0, 0, 0]
    elif compare_strings(age, 'Child'):
        age = [1, 1, 1, 0, 0]
    elif compare_strings(age, 'Adolescent'):
        age = [1, 1, 1, 1, 0]
    elif compare_strings(age, 'Young Adult'):
        age = [1, 1, 1, 1, 1]
    ageonehot = np.array(age)

    valves = np.array(get_locations(current_patient_data))
    templ = np.zeros(4)
    for i, l in enumerate(['MV', 'AV', 'PV', 'TV']):
        picks = np.where(valves == l)[0]
        if len(picks) > 0:
            templ[i] = 1

    return np.concatenate([sex1hot, pregnant1hot, ageonehot, templ])


class TrainDataset(Dataset):
    """ Zweiter dataloader Random ableitung pro patient """

    def __init__(self, patient_files, data_folder, transform=None, mode="train"):
        assert mode in ["train", "val", "test", "threshold"]
        self.data_folder = data_folder
        self.transform = transform
        self.mode = mode
        self.patient_files = patient_files

    def __len__(self):
        return len(self.patient_files)

    def to_label(self, st):
        murmur_classes = ['Present', 'Unknown', 'Absent']
        outcome_classes = ['Abnormal', 'Normal']

        if st in murmur_classes:
            label = murmur_classes.index(st)
        elif st in outcome_classes:
            label = outcome_classes.index(st)
        return label

    def get_target_from_patient(self, idx):
        current_patient_data = load_patient_data(self.patient_files[idx])
        # print("most audible", most_audible_loc)

        # get labels
        label = self.to_label(helper_code.get_murmur(current_patient_data))
        label2 = self.to_label(helper_code.get_outcome(current_patient_data))
        target = {"murmur": label, "outcome": label2}

        # one hot encode
        mumur_1hot_pat1 = np.zeros(3, dtype=float)
        mumur_1hot_pat1[target["murmur"]] = 1
        target["murmur"] = mumur_1hot_pat1

        # outcome
        outcome_1hot_pat1 = np.zeros(2, dtype=float)
        outcome_1hot_pat1[target["outcome"]] = 1
        target["outcome"] = outcome_1hot_pat1

        # wo hörbar (sigmoid)
        hearable_locs = get_hearable_leads(current_patient_data)  # can be nan, MV, PV, ...
        wo_hörbar = np.zeros(4)
        for loc in hearable_locs:
            if loc == "nan" or loc == "Phc":
                continue
            wo_hörbar[['MV', 'AV', 'PV', 'TV'].index(loc)] = 1
        target["wo_hörbar"] = wo_hörbar

        murmur_timing, _ = get_murmur_timing(current_patient_data)
        murmur_shape, _ = get_murmur_shape(current_patient_data)
        murmur_grading, _ = get_murmur_grading(current_patient_data)
        murmur_pitch, _ = get_murmur_pitch(current_patient_data)
        murmur_quality, _ = get_murmur_quality(current_patient_data)
        target["aux_stuff"] = [murmur_timing, murmur_shape, murmur_grading, murmur_pitch, murmur_quality]

        return target

    def get_patient(self, patient_idx):
        current_patient_data = load_patient_data(self.patient_files[patient_idx])
        current_recordings = load_recordings(self.data_folder, current_patient_data)
        valves = np.array(get_locations(current_patient_data))

        img_list = []
        for l in ['MV', 'AV', 'PV', 'TV']:
            picks = np.where(valves == l)[0]
            if len(picks) > 0:
                pick = random.choice(picks) if self.mode == "train" else picks[0]
                img = generate_crops_from_lead(current_recordings, self.transform, self.mode, pick)
            else:
                n_crops = N_CROPS_IN_TRAIN if self.mode == "train" else N_CROPS_IN_VAL
                img = np.zeros((n_crops, CROP_Y, CROP_X, 3))
            img_list.append(img)

        target = self.get_target_from_patient(patient_idx)
        tab = get_tab_from_patient_data(load_patient_data(self.patient_files[patient_idx]))  # , self.data_folder

        return img_list, target, tab

    def mix_target(self, target, target2, lead):
        # print("Mixing", lead)

        target["wo_hörbar"][['MV', 'AV', 'PV', 'TV'].index(lead)] = target2["wo_hörbar"][
            ['MV', 'AV', 'PV', 'TV'].index(lead)]
        target["murmur"] = target["murmur"] * 0.75 + target2["murmur"] * 0.25
        target["outcome"] = target["outcome"] * 0.75 + target2["outcome"] * 0.25
        for i in range(5):
            target["aux_stuff"][i] = target["aux_stuff"][i] * 0.75 + target2["aux_stuff"][i] * 0.25

        return target

    def mix_lead(self, img_list, img_list2, lead):
        replace_idx = ['MV', 'AV', 'PV', 'TV'].index(lead)
        img_list[replace_idx] = img_list2[replace_idx]
        return img_list

    def mix_tab(self, tab, tab2):
        tab[0:2] = tab[0:2] * 0.75 + tab2[0:2] * 0.25
        tab[2:4] = tab[2:4] * 0.75 + tab2[2:4] * 0.25
        tab[4:9] = tab[4:9] * 0.75 + tab2[4:9] * 0.25
        tab[9:13] = tab[9:13] * 0.75 + tab2[9:13] * 0.25
        return tab

    def __getitem__(self, patient_idx):
        img_list, target, tab = self.get_patient(patient_idx)
        if self.mode == "train" and random.random() > 0.2:
            p2_pick = random.randint(0, self.__len__() - 1)
            img_list2, target2, tab2 = self.get_patient(p2_pick)
            choices = np.unique(np.random.choice(['MV', 'AV', 'PV', 'TV'], 1))  # todo: currently only works for 1 lead
            for choice in choices:
                img_list = self.mix_lead(img_list, img_list2, choice)
                target = self.mix_target(target, target2, choice)
                tab = self.mix_tab(tab, tab2)
        img_list = torch.tensor(np.array(img_list)).float()
        tab = torch.tensor(tab).float()
        return img_list, tab, target
