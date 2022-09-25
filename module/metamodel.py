from module.data import TrainDataset, train_transform, val_transform, get_tab_from_patient_data, \
    generate_crops_from_lead
from module.model import ResNet
from module.dataframe import get_df_from_patients, add_folds
from helper_code import find_patient_files, get_locations
import numpy as np
from torch.utils.data import DataLoader, Dataset
from module.engines import Fitter
from module.config import TrainGlobalConfig
import glob
import torch
from tqdm.auto import tqdm
config = TrainGlobalConfig()

from module.threshold_search import generate_matrix
from sklearn.linear_model import LogisticRegression
from module.threshold_search import iterate_loader
from evaluate_model import compute_cost, compute_weighted_accuracy, compute_accuracy, compute_f_measure
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from module.threshold_search import *

def train_metamodel(df, fitter, patients):


    # make 10 different splits for patients
    a = patients[0:len(patients)//2]
    b = patients[len(patients)//2:]
    train_ds = TrainDataset(a, data_folder, transform=val_transform, mode="val")
    thresh_ds = TrainDataset(b, data_folder, transform=val_transform, mode="val")




    #murmur_probas, murmur_labels = iterate_loader(loader, fitter)
    #print("murmur_probas", murmur_probas.shape, "murmur_labels", murmur_labels.shape)
    #print(murmur_probas[0], murmur_labels[0])# [0.33193481 0.38066784 0.28739733] [0. 0. 1.]

    fitter.model.eval()

    y = []
    X_probas = []
    for step, (images, tab, target) in tqdm(enumerate(loader)):
        images = images.to(fitter.device)
        input2 = tab.to(fitter.device)

        y_murmur = target["murmur"].detach().cpu().numpy().tolist()
        y_outcome = target["outcome"].detach().cpu().numpy().tolist()

        assert len(images) == 1

        with torch.no_grad():
            out = fitter.model(images, input2)

        pred_murmur = out[0].softmax(axis=1).detach().cpu().numpy()
        pred_outcome = out[1].softmax(axis=1).detach().cpu().numpy()
        pred_woHörbar = torch.sigmoid(out[2]).detach().cpu().numpy()
        pred_aux1 = out[3].softmax(axis=1).detach().cpu().numpy()
        pred_aux2 = out[4].softmax(axis=1).detach().cpu().numpy()
        pred_aux3 = out[4].softmax(axis=1).detach().cpu().numpy()
        pred_aux4 = out[5].softmax(axis=1).detach().cpu().numpy()
        pred_aux5 = out[6].softmax(axis=1).detach().cpu().numpy()
        pred_aux6 = out[7].softmax(axis=1).detach().cpu().numpy()

        #for j in [pred_murmur, pred_outcome, pred_woHörbar, pred_aux1, pred_aux2, pred_aux3, pred_aux4, pred_aux5, pred_aux6]:
        #    print(j.shape)
        preds = np.concatenate([pred_murmur, pred_outcome, pred_woHörbar, pred_aux1, pred_aux2, pred_aux3, pred_aux4, pred_aux5, pred_aux6], axis=1)[0]

        X_probas.append(preds)
        y.extend(y_murmur)
    y = np.array(y).argmax(axis=1)
    X_probas = np.array(X_probas)

    clf = LogisticRegression(random_state=0, multi_class="ovr")
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(clf, X_probas, y, scoring='accuracy', cv=cv, n_jobs=-1)
    print('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))


    clf = LogisticRegression(random_state=0, multi_class="ovr").fit(X_probas, y)
    y = []
    X_probas = []
    for step, (images, tab, target) in tqdm(enumerate(test_loader)):
        images = images.to(fitter.device)
        input2 = tab.to(fitter.device)

        y_murmur = target["murmur"].detach().cpu().numpy().tolist()
        y_outcome = target["outcome"].detach().cpu().numpy().tolist()

        assert len(images) == 1

        with torch.no_grad():
            out = fitter.model(images, input2)

        pred_murmur = out[0].softmax(axis=1).detach().cpu().numpy()
        pred_outcome = out[1].softmax(axis=1).detach().cpu().numpy()
        pred_woHörbar = torch.sigmoid(out[2]).detach().cpu().numpy()
        pred_aux1 = out[3].softmax(axis=1).detach().cpu().numpy()
        pred_aux2 = out[4].softmax(axis=1).detach().cpu().numpy()
        pred_aux3 = out[4].softmax(axis=1).detach().cpu().numpy()
        pred_aux4 = out[5].softmax(axis=1).detach().cpu().numpy()
        pred_aux5 = out[6].softmax(axis=1).detach().cpu().numpy()
        pred_aux6 = out[7].softmax(axis=1).detach().cpu().numpy()

        #for j in [pred_murmur, pred_outcome, pred_woHörbar, pred_aux1, pred_aux2, pred_aux3, pred_aux4, pred_aux5, pred_aux6]:
        #    print(j.shape)
        preds = np.concatenate([pred_murmur, pred_outcome, pred_woHörbar, pred_aux1, pred_aux2, pred_aux3, pred_aux4, pred_aux5, pred_aux6], axis=1)[0]

        X_probas.append(preds)
        y.extend(y_murmur)
    y = np.array(y)#.argmax(axis=1)
    X_probas = np.array(X_probas)



    pred = clf.predict_proba(X_probas)

    score_matrix = compute_heatmap(pred, y)
    dfx = pd.DataFrame(score_matrix, columns=["absent_threshold", "unknown_threshold", "score"])
    dfx = dfx.pivot("absent_threshold", "unknown_threshold", "score").round(2)
    unknown_threshold, absent_threshold = threshold_from_matrix(dfx)
    print("THRESHOLD FOUND", unknown_threshold, absent_threshold)
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(dfx, annot=True, linewidths=.5, ax=ax)
    plt.show()






    pred = np.argmax(pred, axis=1)








    pred_list = []
    for i in range(len(pred)):
        template = [0, 0, 0]
        template[pred[i]] = 1
        pred_list.append(template)

    pred_list = np.array(pred_list)
    print(y.shape, y[0], pred_list.shape, pred_list[0])

    score = compute_weighted_accuracy(y, pred_list, ['Present', 'Unknown', 'Absent'])
    print("SCORE", score)
    print("SCORE 2", compute_accuracy(y, pred_list))

