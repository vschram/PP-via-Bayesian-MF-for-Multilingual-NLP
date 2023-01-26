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
    #         score_matrix[src_lang][tgt_lang] = score
    score_matrix.fillna(0, inplace=True)
    return score_matrix


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


def get_permutations(a, b):
    hyperparam_settings = []
    h = 0
    for alpha in a:
        for beta_1 in b[0]:
            for beta_2 in b[1]:
                for beta_3 in b[2]:
                    for beta_4 in b[3]:
                        for beta_5 in b[4]:
                            for beta_6 in b[5]:
                                for beta_7 in b[6]:
                                    h += 1
                                    hyperparam = {h: {'alpha': alpha, 'beta': [beta_1, beta_2, beta_3, beta_4, beta_5, beta_6, beta_7]}}
                                    hyperparam_settings.append(hyperparam)

    return hyperparam_settings

def get_median_values(alpha_vec, reg_w_vec, reg_h_vec, reg_bias_s_vec, reg_bias_t_vec, reg_x_vec,reg_y_vec, reg_z_vec):
    median_alpha = np.median(alpha_vec)
    median_reg_w, median_reg_h = np.median(reg_w_vec), np.median(reg_h_vec)
    median_reg_bias_s, median_reg_bias_t = np.median(reg_bias_s_vec), np.median(reg_bias_t_vec)

    try:
        median_reg_x = np.median(reg_x_vec)
    except:
        median_reg_x = None
    try:
        median_reg_y = np.median(reg_y_vec)
    except:
        median_reg_y = None
    try:
        median_reg_z = np.median(reg_z_vec)
    except:
        median_reg_z = None

    return median_alpha, median_reg_w, median_reg_h, median_reg_bias_s, median_reg_bias_t, median_reg_x, median_reg_y, median_reg_z

def create_directory(task, host_name, savepngs, attribute, folds, run, ctx, dim):
    path = f'../results_{host_name}/{task}'
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