import sys
import os
import train_model
import glob

from module.dataframe import get_df_from_patients, place_data, add_folds
from evaluate_model import evaluate_model
from helper_code import find_patient_files

DATASET_PATH = ...  # insert path here


if __name__ == '__main__':
    if not sys.argv[1] in ["train", "inf", "eval", "train5", "traincont"]:
        raise Exception(f"Parsed argument '{sys.argv[1]}' must be either 'train', 'inf', 'eval', 'train5', 'traincont'")

    dataset_path = DATASET_PATH
    df = get_df_from_patients(find_patient_files(dataset_path))
    df = add_folds(df, num_folds=6, random_state=0)

    if sys.argv[1] == "train":
        print("Start training")
        for f in glob.glob('./model/*'):
            os.remove(f)
        """for f in glob.glob('./training_data/*'):
            os.remove(f)
        for f in glob.glob('./test_data/*'):
            os.remove(f)"""
        place_data(df, dataset_path=dataset_path, train_folder='./training_data/', test_folder='./test_data/', test_fold=1)  # 99 means no patients in test
        os.system("python train_model.py './training_data/' './model/'")

    elif sys.argv[1] == "inf":
        print("Inference")
        place_data(df, dataset_path=dataset_path, train_folder='./training_data/', test_folder='./test_data/',
                   test_fold=1)
        for f in glob.glob('./test_outputs/*'):
            os.remove(f)
        os.system(f"python run_model.py './model/' './test_data/' './test_outputs/'")  # {sys.argv[2]}

    elif sys.argv[1] == "eval":
        print("Evaluation")
        out = evaluate_model('./test_data/', './test_outputs/')
        print("Murmur w_acc:", out[0][9])
        print("Outcome cost", out[1][-1])

    elif sys.argv[1] == "train5":
        print("Train and Eval on 5 test folds, only for local test purposes")
        arr = []
        arr2 = []
        for i in range(5):
            for f in glob.glob('./model/*'):
                os.remove(f)
            for f in glob.glob('./training_data/*'):
                os.remove(f)
            for f in glob.glob('./test_data/*'):
                os.remove(f)
            place_data(df, dataset_path=dataset_path, train_folder='./training_data/', test_folder='./test_data/', test_fold=i)
            os.system("python train_model.py './training_data/' './model/'")
            for f in glob.glob('./test_outputs/*'):
                os.remove(f)
            os.system(f"python run_model.py './model/' './test_data/' './test_outputs/'")
            out = evaluate_model('./test_data/', './test_outputs/')
            print("_"*50)
            print("OVERALLSCORE:", out[0][9], "Cost:", out[1][-1])
            print("_"*50)
            arr.append(out[0][9])
            arr2.append(out[1][-1])
        print("5 folds 5 accs:", arr)
        print("5 folds 5 costs:", arr2)

    elif sys.argv[1] == "traincont":
        print("Start training")
        #place_data(df, dataset_path=dataset_path, train_folder='../training_data/', test_folder='../test_data/', test_fold=4)  # 99 means no patients in test
        os.system("python train_model.py './training_data/' './model/'")




# python train_model.py "../training_data/" "../model/"
# python run_model.py "../model/" "../test_data/" "../test_outputs/"
# python evaluate_model.py "../test_data/" "../test_outputs/"
