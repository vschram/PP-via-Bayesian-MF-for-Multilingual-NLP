import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

def split_k_fold_data(file_dir, score_index_name, src_index_name, tgt_index_name, k=5):
    data = pd.read_csv(file_dir[1])

    # shuffle
    data = data.sample(frac=1)

    # generate score matrix
    src_langs = data[src_index_name].unique()
    tgt_langs = data[tgt_index_name].unique()
    score_matrix = pd.DataFrame(index=src_langs, columns=tgt_langs, data=np.nan)

    # eliminate empty rows and columns
    data = data.dropna(axis=1, how="all")
    data = data.dropna(axis=0, how="all")

    # K fold split
    k_fold_data = {}
    models = list(score_index_name)
    lens = len(data)

    for i, model in enumerate(models):
        ex_per_fold = int(np.ceil(lens / k))
        for j in range(k):
            start = ex_per_fold * j
            end = ex_per_fold * (j + 1)
            if j == 0:
                k_fold_data[model] = {"train": [], "test": []}
            k_fold_data[model]["train"].append(pd.concat([data.iloc[:start, :], data.iloc[end:, :]], axis=0))
            k_fold_data[model]["test"].append(data.iloc[start:end, :])
    return k_fold_data, score_matrix

def split_k_fold_data_ncv(file_dir, score_index_name, src_index_name, tgt_index_name, k=5):
    data_orig = pd.read_csv(file_dir[1])
    data = data_orig.copy()
    data['BLEU'] = data['BLEU']/100

    # shuffle
    data = data.sample(frac=1)

    # generate score matrix
    src_langs = data[src_index_name].unique()
    tgt_langs = data[tgt_index_name].unique()
    score_matrix = pd.DataFrame(index=src_langs, columns=tgt_langs, data=np.nan)

    # eliminate empty rows and columns
    data = data.dropna(axis=1, how="all")
    data = data.dropna(axis=0, how="all")

    # K fold split
    k_fold_data = {}
    models = list(score_index_name)
    lens = len(data)

    for i, model in enumerate(models):
        ex_per_fold = int(np.ceil(lens / k))
        for j in range(k):
            start = ex_per_fold * j
            end = ex_per_fold * (j + 1)
            if j == 0:
                k_fold_data[model] = {"train": [], "test": [], "full_train": []}
            k_fold_data[model]["train"].append(pd.concat([data.iloc[:start, :], data.iloc[end:, :]], axis=0))
            k_fold_data[model]["full_train"].append(pd.concat([data.iloc[:start, :], data.iloc[end:, :]], axis=0))

            # dataset split for nested CV, for hyperparam optimization
            # --------------------------------------------------------
            # --------------------------------------------------------
            cv_data = k_fold_data[model]["train"][j]
            cv_lens = len(cv_data)
            cv_ex_per_fold = int(np.ceil(cv_lens / (k-1)))
            for jj in range(k-1):
                cv_start = cv_ex_per_fold * jj
                cv_end = cv_ex_per_fold * (jj + 1)
                if jj == 0:
                    k_fold_data[model]["train"][j] = {"train": [], "dev": []}
                k_fold_data[model]["train"][j]["train"].append(pd.concat([cv_data.iloc[:cv_start, :], cv_data.iloc[cv_end:, :]], axis=0))
                k_fold_data[model]["train"][j]["dev"].append(cv_data.iloc[cv_start:cv_end, :])
            # --------------------------------------------------------
            # --------------------------------------------------------

            k_fold_data[model]["test"].append(data.iloc[start:end, :])
    return k_fold_data, score_matrix


def split_k_fold_data_nlperf(file_dir, score_index_name, src_index_name, tgt_index_name, nr, k=5):

    filename = f'{file_dir[0]}{nr+1}.pkl'
    with open(filename, 'rb') as f:
        split_data_prepared = pickle.load(f)
        f.close()

    data = pd.read_csv(file_dir[1])

    # generate score matrix
    src_langs = data[src_index_name].unique()
    tgt_langs = data[tgt_index_name].unique()
    score_matrix = pd.DataFrame(index=src_langs, columns=tgt_langs, data=np.nan)


    # K fold split
    k_fold_data = {}
    models = list(score_index_name)

    for i, model in enumerate(models):
        for j in range(k):
            if j == 0:
                k_fold_data[model] = {"train": [], "test": [], "full_train": []}
            split_data_prepared[model]["train_langs"][j].reset_index(drop=True, inplace=True)
            split_data_prepared[model]["train_labels"][j].reset_index(drop=True, inplace=True)
            split_data_prepared[model]["train_feats"][j].reset_index(drop=True, inplace=True)
            k_fold_data[model]["train"].append(pd.concat([split_data_prepared[model]["train_langs"][j], split_data_prepared[model]["train_labels"][j], split_data_prepared[model]["train_feats"][j]], axis=1))
            k_fold_data[model]["full_train"].append(pd.concat([split_data_prepared[model]["train_langs"][j], split_data_prepared[model]["train_labels"][j], split_data_prepared[model]["train_feats"][j]], axis=1))

            # dataset split for nested CV, for hyperparam optimization
            # --------------------------------------------------------
            # --------------------------------------------------------
            cv_data = k_fold_data[model]["train"][j]
            cv_lens = len(cv_data)
            cv_ex_per_fold = int(np.ceil(cv_lens / (k - 1)))
            for jj in range(k - 1):
                cv_start = cv_ex_per_fold * jj
                cv_end = cv_ex_per_fold * (jj + 1)
                if jj == 0:
                    k_fold_data[model]["train"][j] = {"train": [], "dev": []}
                k_fold_data[model]["train"][j]["train"].append(
                    pd.concat([cv_data.iloc[:cv_start, :], cv_data.iloc[cv_end:, :]], axis=0))
                k_fold_data[model]["train"][j]["dev"].append(cv_data.iloc[cv_start:cv_end, :])
            # --------------------------------------------------------
            # --------------------------------------------------------
            split_data_prepared[model]["test_langs"][j].reset_index(drop=True, inplace=True)
            split_data_prepared[model]["test_labels"][j].reset_index(drop=True, inplace=True)
            split_data_prepared[model]["test_feats"][j].reset_index(drop=True, inplace=True)
            k_fold_data[model]["test"].append(pd.concat([split_data_prepared[model]["test_langs"][j], split_data_prepared[model]["test_labels"][j], split_data_prepared[model]["test_feats"][j]], axis=1))

    return k_fold_data, score_matrix