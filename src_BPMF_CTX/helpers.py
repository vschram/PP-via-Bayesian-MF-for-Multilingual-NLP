import numpy as np
from copy import deepcopy
import os

def generate_score_matrix(train_data, src_index_name, tgt_index_name, score_index_name, origin_score_matrix):
    score_matrix = deepcopy(origin_score_matrix)

    for record in train_data.iterrows():
        record = record[1]
        src_lang = record[src_index_name]
        tgt_lang = record[tgt_index_name]
        score = record[score_index_name]
        score_matrix.loc[src_lang, tgt_lang] = score

    return score_matrix

def generate_feature_matrix(train_data, src_index_name, tgt_index_name, score_index_name, origin_score_matrix, lang_pair_side_info, lang_pair_si_len):
    feat_mat = []
    for i in range(0, lang_pair_si_len, 1):
        score_matrix = deepcopy(origin_score_matrix)
        for record in train_data.iterrows():
            record = record[1]
            src_lang = record[src_index_name]
            tgt_lang = record[tgt_index_name]
            if src_lang + '_' + tgt_lang == 'wbp_bxr':
                a = 0
            feature = lang_pair_side_info[src_lang + '_' + tgt_lang]
            score = feature[i]
            score_matrix.loc[src_lang, tgt_lang] = score
        feat_mat.append(score_matrix)

    return feat_mat


def get_rmse(valid_data, model, src_index_name, tgt_index_name, score_matrix, train_matrix):
    rmse = 0.0
    src_langs = train_matrix.index.tolist()
    tgt_langs = train_matrix.columns.tolist()
    for cur_valid_data in valid_data.iterrows():
        cur_valid_data = cur_valid_data[1]
        src_lang, tgt_lang, score = cur_valid_data[src_index_name], cur_valid_data[tgt_index_name], cur_valid_data[
            model]
        src_idx = src_langs.index(src_lang)
        tgt_idx = tgt_langs.index(tgt_lang)
        prediction = score_matrix[src_idx][tgt_idx]
        rmse += (prediction - score) * (prediction - score)
    return np.sqrt(rmse / len(valid_data))


def get_permutations(a):
    hyperparam_settings = []
    h = 0
    for alpha in a:
        h+= 1
        hyperparam = {h: {'alpha': alpha}}
        hyperparam_settings.append(hyperparam)

    return hyperparam_settings

def get_median_values(alpha_vec):
    median_alpha = np.median(alpha_vec)

    return median_alpha

def create_directory(task, host_name, savepngs, attribute, folds, run, ctx, dim, model):
    path = f'../results_{host_name}/{task}/model_{model}'
    if not os.path.exists(path):
        os.makedirs(path)

    if savepngs:
        for i in range(folds):
            for j in range(folds-1):
                for r in range(run):
                    pngpaths = f'{path}/dim_{dim}/attribute_{attribute}/ctx_number_{ctx}/run_{r+1}/fold_{i+1}/cv_fold_{j+1}'

                    if not os.path.exists(pngpaths):
                        os.makedirs(pngpaths)

    pathlogs = f'{path}/dim_{dim}/attribute_{attribute}/ctx_number_{ctx}/logs'
    if not os.path.exists(pathlogs):
        os.makedirs(pathlogs)

    return path, pathlogs

def get_data(task, split):
    if split == 'NCV_NLPerf':
        split_dir = f"../data/NLPerfSplit/{task}_split_"
        data_dir = f"../data/data_{task}.csv"
    else:
        split_dir = f"../data/data_{task}.csv"
        data_dir = f"../data/data_{task}.csv"

    dir = [split_dir, data_dir]

    return dir

def create_dict_for_results(task):
    if task == "wiki_MF" or task == "wiki_aaMF" or task == "tsfmt_MF" or task == "tsfmt_aaMF":
        results_all = {"BLEU": {"run_1": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                 "run_2": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                 "run_3": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                 "run_4": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                 "run_5": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                 "run_6": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                 "run_7": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                 "run_8": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                 "run_9": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                 "run_10": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}}}}
    elif task == "bli_MF" or task == "bli_aaMF":
        results_all = {"Vecmap": {"run_1": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                 "run_2": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                 "run_3": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                 "run_4": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                 "run_5": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                 "run_6": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                 "run_7": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                 "run_8": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                 "run_9": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                 "run_10": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}}},
                       "Muse": {"run_1": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                  "run_2": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_2":{"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                  "run_3": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                  "run_4": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                  "run_5": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                  "run_6": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                  "run_7": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                  "run_8": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                  "run_9": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                  "run_10": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0}, "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}}}
                       }

    elif task == "tsfel_MF" or task == "tsfel_aaMF" or task == "tsfpos_MF" or task == "tsfpos_aaMF":
        results_all = {"Accuracy": {"run_1": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                "run_2": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                "run_3": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                "run_4": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                "run_5": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                "run_6": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                "run_7": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                "run_8": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                "run_9": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                          "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                "run_10": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                           "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                           "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                           "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                           "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0}}}}

    elif task == "tsfparsing_MF" or task == "tsfparsing_aaMF":
        results_all = {"Accuracy": {"run_1": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_6": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_7": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_8": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_9": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_10": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_11": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_12": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_13": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_14": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_15": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_16": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_17": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_18": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_19": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_20": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_21": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_22": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_23": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_24": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_25": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_26": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_27": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_28": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_29": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_30": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                    "run_2": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_6": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_7": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_8": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_9": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_10": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_11": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_12": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_13": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_14": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_15": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_16": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_17": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_18": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_19": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_20": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_21": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_22": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_23": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_24": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_25": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_26": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_27": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_28": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_29": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_30": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                    "run_3": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_6": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_7": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_8": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_9": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_10": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_11": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_12": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_13": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_14": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_15": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_16": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_17": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_18": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_19": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_20": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_21": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_22": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_23": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_24": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_25": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_26": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_27": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_28": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_29": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_30": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                    "run_4": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_6": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_7": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_8": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_9": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_10": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_11": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_12": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_13": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_14": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_15": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_16": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_17": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_18": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_19": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_20": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_21": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_22": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_23": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_24": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_25": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_26": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_27": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_28": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_29": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_30": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                    "run_5": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_6": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_7": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_8": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_9": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_10": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_11": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_12": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_13": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_14": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_15": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_16": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_17": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_18": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_19": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_20": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_21": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_22": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_23": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_24": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_25": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_26": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_27": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_28": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_29": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_30": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                    "run_6": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_6": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_7": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_8": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_9": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_10": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_11": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_12": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_13": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_14": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_15": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_16": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_17": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_18": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_19": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_20": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_21": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_22": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_23": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_24": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_25": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_26": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_27": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_28": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_29": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_30": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                    "run_7": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_6": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_7": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_8": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_9": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_10": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_11": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_12": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_13": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_14": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_15": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_16": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_17": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_18": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_19": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_20": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_21": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_22": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_23": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_24": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_25": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_26": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_27": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_28": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_29": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_30": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                    "run_8": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_6": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_7": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_8": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_9": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_10": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_11": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_12": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_13": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_14": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_15": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_16": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_17": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_18": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_19": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_20": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_21": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_22": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_23": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_24": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_25": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_26": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_27": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_28": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_29": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_30": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                    "run_9": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_6": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_7": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_8": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_9": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_10": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_11": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_12": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_13": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_14": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_15": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_16": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_17": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_18": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_19": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_20": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_21": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_22": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_23": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_24": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_25": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_26": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_27": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_28": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_29": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                              "fold_30": {"results": 0, "trace": 0, "source": 0, "target": 0}},
                                    "run_10": {"fold_1": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                               "fold_2": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                               "fold_3": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                               "fold_4": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                               "fold_5": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                               "fold_6": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                               "fold_7": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                               "fold_8": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                               "fold_9": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                               "fold_10": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                               "fold_11": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                               "fold_12": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                               "fold_13": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                               "fold_14": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                               "fold_15": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                               "fold_16": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                               "fold_17": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                               "fold_18": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                               "fold_19": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                               "fold_20": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                               "fold_21": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                               "fold_22": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                               "fold_23": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                               "fold_24": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                               "fold_25": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                               "fold_26": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                               "fold_27": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                               "fold_28": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                               "fold_29": {"results": 0, "trace": 0, "source": 0, "target": 0},
                                               "fold_30": {"results": 0, "trace": 0, "source": 0, "target": 0}}}}

    return results_all