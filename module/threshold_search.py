
import pandas as pd
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from evaluate_model import find_challenge_files, load_murmurs, load_classifier_outputs
from evaluate_model import compute_cost, compute_weighted_accuracy, compute_accuracy, compute_f_measure
from evaluate_model import compute_auc, enforce_positives, load_outcomes



TTA = False


def iterate_loader(thresh_loader, fitter, only=None):
    fitter.model.eval()

    murmur_labels = []
    murmur_probas = []
    outcome_labels = []
    outcome_probas = []

    for step, (images, input2, target) in enumerate(thresh_loader):
        images = images.to(fitter.device)
        input2 = input2.to(fitter.device)

        y_murmur = target["murmur"].detach().cpu().numpy().tolist()
        y_outcome = target["outcome"].detach().cpu().numpy().tolist()

        batch_size = len(images)

       # ispregnant = True if input2[0][3] == 1 else False
       # sex = "female" if input2[0][0] == 1 else "male"

        if TTA == True:
            images = torch.concat([images for i in range(5)])
            for i in range(4):
                images[i][i] = torch.zeros_like(images[i][i])

        with torch.no_grad():
            out = fitter.model(images, input2)
        probas_murmur = torch.max(out[0], dim=0)[0].unsqueeze(0)
        probas_outcome = torch.max(out[1], dim=0)[0].unsqueeze(0)

        # pred_murmur = torch.sigmoid(probas_murmur).detach().cpu().numpy().tolist()
        # pred_outcome = torch.sigmoid(probas_outcome).detach().cpu().numpy().tolist()

        pred_murmur = probas_murmur.softmax(axis=1).detach().cpu().numpy().tolist()
        pred_outcome = probas_outcome.softmax(axis=1).detach().cpu().numpy().tolist()

        """if only is not None:
            if only == "woman":
                if ispregnant == False and sex == "female":
                    pass
                else:
                    continue
            if only == "man":
                if sex == "male":
                    pass
                else:
                    continue
            if only == "pregnant":
                if ispregnant == True:
                    pass
                else:
                    continue"""

        murmur_probas.extend(pred_murmur)
        murmur_labels.extend(y_murmur)
        outcome_probas.extend(pred_outcome)
        outcome_labels.append(y_outcome)

    murmur_labels = np.array(murmur_labels)
    murmur_probas = np.array(murmur_probas)
    outcome_labels = np.array(outcome_labels).squeeze()
    outcome_probas = np.array(outcome_probas)
    return murmur_probas, murmur_labels, outcome_probas, outcome_labels


def compute_heatmap(nn_probas, labels):
    score_matrix = []
    for absent_threshold in np.arange(0.0, 1.05, 0.05):
        absent_threshold = round(absent_threshold, 2)
        for unknown_threshold in np.arange(0.0, 1.05, 0.05):
            unknown_threshold = round(unknown_threshold, 2)

            nn_probas_binary = np.zeros(nn_probas.shape, dtype=int)

            nn_probas_binary[:] = [1, 0, 0]

            nn_probas_binary[nn_probas[:, 2] > absent_threshold] = [0, 0, 1]
            nn_probas_binary[nn_probas[:, 1] > unknown_threshold] = [0, 1, 0]

            nn_probas_binary = enforce_positives(nn_probas_binary, ['Present', 'Unknown', 'Absent'], 'Present')

            score = compute_weighted_accuracy(labels, nn_probas_binary, ['Present', 'Unknown', 'Absent'])
            score_matrix.append([absent_threshold, unknown_threshold, score])
    return score_matrix


def threshold_from_matrix(dfx):
    template = np.zeros(dfx.values.shape)
    template[dfx.values == np.max(dfx.values)] = 1
    absent_idx, unknown_idx = np.where(template)
    absent_mid = (min(absent_idx) + max(absent_idx)) // 2
    unknown_mid = (min(unknown_idx) + max(unknown_idx)) // 2
    unknown_threshold = dfx.T.index[unknown_mid]
    absent_threshold = dfx.index[absent_mid]

    print(unknown_threshold, absent_threshold)
    return unknown_threshold, absent_threshold
import time


def generate_matrix(thresh_loader, fitter, innerfold, model_nr):
    threshold_list = []

    """fig, ax = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(30, 7))
    for i, what in enumerate(["woman", "man", "pregnant", None]):

        murmur_probas, murmur_labels = iterate_loader(thresh_loader, fitter, only=what)
        score_matrix = compute_heatmap(murmur_probas, murmur_labels)

        dfx = pd.DataFrame(score_matrix, columns=["absent_threshold", "unknown_threshold", "score"])
        dfx = dfx.pivot("absent_threshold", "unknown_threshold", "score").round(2)
        unknown_threshold, absent_threshold = threshold_from_matrix(dfx)

        threshold_list.append([unknown_threshold, absent_threshold])
        if what is not None:
            sns.heatmap(dfx, annot=True, linewidths=.5, ax=ax[i])

    plt.tight_layout()
    plt.show()"""

    murmur_probas, murmur_labels, outcome_probas, outcome_labels = iterate_loader(thresh_loader, fitter, only=None)

    # MURMUR
    score_matrix = compute_heatmap(murmur_probas, murmur_labels)

    dfx = pd.DataFrame(score_matrix, columns=["absent_threshold", "unknown_threshold", "score"])
    dfx = dfx.pivot("absent_threshold", "unknown_threshold", "score").round(2)
    unknown_threshold, absent_threshold = threshold_from_matrix(dfx)

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(dfx, annot=True, linewidths=.5, ax=ax)
    plt.savefig(f'{fitter.base_dir}/hist_model{model_nr}_fold0_{innerfold}.png', dpi=144)
    plt.close(fig)

    # OUTCOME
    costs = []
    for thr in np.arange(0.0, 1.05, 0.05):
        labelll = []
        for i in range(len(outcome_probas)):
            if outcome_probas[i][0] > thr:
                labelll.append([1, 0])
            else:
                labelll.append([0, 1])
        outcome_labels = np.array(outcome_labels).astype(int)
        cost = compute_cost(outcome_labels, np.array(labelll), ["Abnormal", "Normal"], ["Abnormal", "Normal"])
        costs.append(cost)
    best_thresh = np.arange(0.0, 1.05, 0.05)[np.argmin(costs)]

    return [[unknown_threshold, absent_threshold], [best_thresh]], dfx