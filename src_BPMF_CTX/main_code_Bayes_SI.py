##################################################
### Code based on ThianziLi and NLPerf and PMF homepage
#################################################

import os
os.environ['OPENBLAS_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['OMP_NUM_THREADS'] = '16'

import numpy as np
import pickle as pkl
import random
from utils_Bayes import parse_args
import socket
from helpers import generate_score_matrix, get_rmse, get_permutations, get_median_values, create_directory, get_data, create_dict_for_results, generate_feature_matrix
from get_information import get_info, get_context, get_filename
from split_functions import split_k_fold_data, split_k_fold_data_ncv, split_k_fold_data_nlperf, split_k_fold_data_lolo_source, split_k_fold_data_lolo_target
from logger import create_logger
from BPMF_SI import BPMF
from analysis import analyse_rmse_mcmc, analyse_summary, analyse_mcmc
#import seaborn as sns
#import matplotlib.pyplot as plt

def get_result(filename, path, logger, other_params, dim, alpha, data_dir, scores, src_index, tgt_index, k, num_running, std, src_lang_side_info=None,
               tgt_lang_side_info=None, lang_pair_side_info=None, src_si_len=0, tgt_si_len=0,
               lang_pair_si_len=0):

    hyperparam_settings = get_permutations(alpha)

    if other_params[0]:
        logger.info(f'hyperparameters:')
        logger.info(f'ALPHA (precision) = {alpha}')
        logger.info(f'leading to... {len(hyperparam_settings)} different hyperparameter settings')
        logger.info('*'*20)

    if other_params[12]:
        results_all = create_dict_for_results(task)

    # Record best hyperparam settings for each run
    # -----------------------------------------------------
    avg_running_rmse, avg_running_dev_rmse, avg_running_train_rmse = [], [], []

    run_alpha_vec = []

    for nr in range(num_running):
        if other_params[0]:
            logger.info('+' * 50)
            logger.info(f'Starting run {nr + 1}')
            logger.info('+' * 50)

        if other_params[3] == 'CV':
            data, langs_matrix = split_k_fold_data(data_dir, scores, src_index, tgt_index, k)
        elif other_params[3] == 'NCV':
            data, langs_matrix = split_k_fold_data_ncv(data_dir, scores, src_index, tgt_index, k)
        elif other_params[3] == 'NCV_NLPerf':
            data, langs_matrix = split_k_fold_data_nlperf(data_dir, scores, src_index, tgt_index, nr, k)
        elif other_params[3] == 'NCV_lolo_source':
            data, langs_matrix = split_k_fold_data_lolo_source(data_dir, scores, src_index, tgt_index, nr, k)
        elif other_params[3] == 'NCV_lolo_target':
            data, langs_matrix = split_k_fold_data_lolo_target(data_dir, scores, src_index, tgt_index, nr, k)

        model_final_rmse, model_final_rmse_dev, model_final_rmse_train = [], [], []
        model_alpha_vec = []

        if 'lolo_source' in split:
            k = langs_matrix.shape[0]
        elif 'lolo_target' in split:
            k = langs_matrix.shape[1]

        for model in scores:

            if other_params[3] == 'NCV' or other_params[3] == 'NCV_NLPerf' or other_params[3] == 'NCV_lolo_source' or other_params[3] == 'NCV_lolo_target':
                if other_params[0]:
                    logger.info('*' * 20)
                    logger.info(f'Scores: {model} using NCV')
                    logger.info('*' * 20)

                # Average per folds:
                average_rmse, average_rmse_train, average_rmse_dev = [], [], []

                # Record best hyperparam settings for each fold
                # -----------------------------------------------------
                alpha_vec = []

                # Start k-fold
                # -----------------------------------------------------
                for i in range(k):
                    if other_params[0]:
                        logger.info("\tNCV, Fold {}: ".format(i + 1))

                    # Create test set here, but use in every inner CV too, so later no retraining is required.
                    train_data_full, test_data, train_data_complete = data[model]["train"][i], data[model]["test"][i], data[model]["full_train"][i]

                    if len(hyperparam_settings) == 1:

                        argmin_h_dev_error = {'average_dev': 0, 'alpha': hyperparam_settings[0][1]['alpha'], 'h': 1}

                        if other_params[0]:
                            logger.info('*' * 20)
                            logger.info('Only one hyperparam setting, no optimization needed')
                            logger.info('-' * 20)
                            logger.info('Which is always true, because we have BPMF')
                            logger.info('-' * 20)
                            logger.info('*' * 20)

                    else:
                        # --------------------------------------------------
                        # A loop to optimize over hyperparameters
                        # --------------------------------------------------
                        argmin_h_dev_error = {'average_dev': 1000, 'h': 'Not yet found'}

                        for hypset in range(len(hyperparam_settings)):
                            alpha = hyperparam_settings[hypset][hypset+1]['alpha']

                            total_train_rmse_cv, total_dev_rmse_cv, total_rmse_cv = 0.0, 0.0, 0.0

                            if other_params[0]:
                                logger.info('*' * 20)
                                logger.info(f'hyperparam setting: {hypset+1}')
                                logger.info(f'hyperparameters:')
                                logger.info(f'ALPHA (precision) = {alpha}')
                                logger.info('*' * 20)

                            # --------------------------------------------------
                            # Inner CV loop
                            # --------------------------------------------------
                            for ii in range(k-1):
                                train_data, dev_data = train_data_full["train"][ii], train_data_full["dev"][ii]

                                train_matrix = generate_score_matrix(train_data, src_index, tgt_index, model, langs_matrix)
                                dev_matrix = generate_score_matrix(dev_data, src_index, tgt_index, model,
                                                                     langs_matrix)
                                mf = BPMF(model, logger, other_params, train_matrix, dim=dim, alpha=alpha, std=std)

                                # Get train, dev RMSEs
                                # ------------------------------
                                trace = mf.draw_samples(cores=1, draws=other_params[10], tune=other_params[9], chains=other_params[8],
                                                        progressbar=True, return_inferencedata=False)

                                # -----------------------
                                # MAP solution
                                # -----------------------
                                # mf.find_map(method_map)  # "L-BFGS-B"

                                # -----------------------
                                # Evaluate PMF MAP estimates.
                                # -----------------------
                                # test_rmse_map, train_rmse_map, W, H = mf.eval_map(train_data, dev_data)

                                if other_params[0]:
                                    logger.info('*' * 20)
                                    diverging = trace['diverging']
                                    logger.info('Number of Divergent Chains: {}'.format(diverging.nonzero()[0].size))
                                    diverging_pct = diverging.nonzero()[0].size / len(trace) * 100
                                    logger.info('Percentage of Divergent Chains: {:.1f}'.format(diverging_pct))

                                predicted, results, cur_rmse_dev, _ = analyse_rmse_mcmc(logger, other_params, mf, train_matrix, dev_matrix, ii, nr, i, path, alpha, beta)

                                total_dev_rmse_cv += cur_rmse_dev

                                if other_params[0]:
                                    logger.info('*' * 20)
                                    logger.info("\t\t(Inner cv fold {}) dev RMSE is {}.".format(ii+1, cur_rmse_dev))
                                    logger.info("*" * 20)

                            average_dev = total_dev_rmse_cv / (k-1)

                            if argmin_h_dev_error['average_dev'] > average_dev:
                                argmin_h_dev_error = {'average_dev': average_dev, 'alpha': alpha, 'h': hypset+1}

                        if other_params[0]:
                            logger.info("#" * 20)
                            logger.info(f"(Outer fold {i+1}) min average dev rmse, averaged over {ii+1} inner folds: "
                                        + str(argmin_h_dev_error['average_dev'])
                                        + str(argmin_h_dev_error['h']))
                            logger.info("#" * 20)

                    # ----------------------------------------------------------
                    # -- Train model on complete train set with h that causes min dev error
                    # ----------------------------------------------------------
                    other_params[7] = 0
                    train_matrix_full = generate_score_matrix(train_data_complete, src_index, tgt_index, model,
                                                              langs_matrix)
                    feature_matrix_train = generate_feature_matrix(train_data_complete, src_index, tgt_index, model,
                                                       langs_matrix, lang_pair_side_info, lang_pair_si_len)

                    test_matrix = generate_score_matrix(test_data, src_index, tgt_index, model,
                                                       langs_matrix)
                    feature_matrix_test = generate_feature_matrix(test_data, src_index, tgt_index, model,
                                                       langs_matrix, lang_pair_side_info, lang_pair_si_len)

                    mf_full = BPMF(model, logger, other_params, train_matrix_full, src_si_len, tgt_si_len, lang_pair_si_len, dim=dim, alpha=argmin_h_dev_error['alpha'],
                                   std=std,  X=src_lang_side_info, Y=tgt_lang_side_info, Z_orig=feature_matrix_train)

                    trace = mf_full.draw_samples(cores=1, draws=other_params[10], tune=other_params[9],
                                            chains=other_params[8], init="adapt_diag",
                                            progressbar=True, return_inferencedata=False)

                    predicted, results, cur_rmse, cur_rmse_train = analyse_rmse_mcmc(logger, other_params, mf_full,
                                                                                     train_matrix_full, test_matrix, feature_matrix_test, feature_matrix_train, None, nr, i, path, alpha, 1)
                    save_test_data = 0
                    #if save_test_data == 1:
                     #   with open(f'{task}_Model_{model}_test_matrix_split_' + f'{i + 1}' + '.pkl', 'wb') as g:
                      #      pkl.dump(test_matrix, g)
                       # test_matrix = test_matrix.replace(0, np.nan)
                        #sns.set(font_scale=0.7)  # , vmin=0.0, vmax=0.1
                        #fig1 = sns.heatmap(test_matrix, xticklabels=test_matrix.axes[1],
                        #                   yticklabels=test_matrix.axes[0], linewidth=0.5)
                        #fig1.set_title(f'test data, split {i+1}')
                        #fig1.set_xlabel(model)
                        #fig1.set_ylabel(model)

                        #plt.savefig(f'c_{model}_test_data_split_{i+1}.png', bbox_inches='tight')
                        #plt.show()

                        #train_matrix_complete = train_matrix_full
                        #train_matrix_complete = train_matrix_complete.replace(0, np.nan)
                        #sns.set(font_scale=0.7)  # , vmin=0.0, vmax=0.1
                        #fig1 = sns.heatmap(train_matrix_complete, xticklabels=train_matrix_complete.axes[1],
                        #                   yticklabels=train_matrix_complete.axes[0], linewidth=0.5)
                        #fig1.set_title(f'train data, split {i+1}')
                        #fig1.set_xlabel(model)
                        #fig1.set_ylabel(model)

                        #plt.savefig(f'c_{model}_train_data_split_{i+1}.png', bbox_inches='tight')
                        #plt.show()

                    if other_params[12]:
                        results_all[f'{model}'][f"run_{nr + 1}"][f"fold_{i + 1}"]["results"] = results
                        results_all[f'{model}'][f"run_{nr + 1}"][f"fold_{i + 1}"]["trace"] = trace
                        results_all[f'{model}'][f"run_{nr + 1}"][f"fold_{i + 1}"]["source"] = langs_matrix.axes[0]
                        results_all[f'{model}'][f"run_{nr + 1}"][f"fold_{i + 1}"]["target"] = langs_matrix.axes[1]

                    if other_params[0]:
                        logger.info('*' * 20)
                        logger.info("\t\t(Outer Fold {}) train RMSE is {}.".format(i + 1, cur_rmse_train))
                        logger.info("\t\t(Outer Fold {}) test RMSE is {}.".format(i + 1, cur_rmse))
                        logger.info(argmin_h_dev_error)
                        logger.info("*" * 20)
                    #other_params[7] = 1
                    # ----------------------------------------------------------
                    # ----------------------------------------------------------
                    # ----------------------------------------------------------

                    #fold[f'fold {i}'] = argmin_h_dev_error
                    #res[model] = fold
                    average_rmse.append(cur_rmse)
                    average_rmse_train.append(cur_rmse_train)
                    average_rmse_dev.append(argmin_h_dev_error['average_dev'])
                    alpha_vec.append(argmin_h_dev_error['alpha'])

                # find average error here
                # -----------------------------------------
                final_rmse = sum(average_rmse) / k
                final_rmse_std = np.std(average_rmse)
                final_rmse_train = sum(average_rmse_train) / k
                final_rmse_train_std = np.std(average_rmse_train)
                final_rmse_dev = sum(average_rmse_dev) / k
                final_rmse_dev_std = np.std(average_rmse_dev)

                # -----------------------------------------
                # find median hyperparam setting
                # -----------------------------------------
                median_alpha = get_median_values(alpha_vec)

                if other_params[0]:
                    logger.info('#################################')
                    logger.info(f'-----per K-folds for {model}-----')
                    logger.info(f'final_rmse: {final_rmse}')
                    logger.info(f'final_rmse_std: {final_rmse_std}')
                    logger.info(f'--------------------------------')
                    logger.info(f'final_rmse_train: {final_rmse_train}')
                    logger.info(f'final_rmse_train_std: {final_rmse_train_std}')
                    logger.info(f'--------------------------------')
                    logger.info(f'final_rmse_dev: {final_rmse_dev}')
                    logger.info(f'final_rmse_dev_std: {final_rmse_dev_std}')
                    logger.info(f'--------------------------------')
                    logger.info(f'median_alpha: {median_alpha}')
                    logger.info('#################################')

            elif other_params[3] == 'CV':
                print('TODO')


            # -----------------------------------------
            # Average per model here
            # -----------------------------------------
            model_final_rmse.append(final_rmse)
            model_final_rmse_train.append(final_rmse_train)
            model_final_rmse_dev.append(final_rmse_dev)

            model_alpha_vec.append(median_alpha)

        # -----------------------------------------
        # find median hyperparam setting aver all models
        # -----------------------------------------
        median_alpha = get_median_values(model_alpha_vec)
        # -----------------------------------------
        # Average per model here
        # -----------------------------------------
        avg_running_rmse.append(sum(model_final_rmse)/len(scores))
        avg_running_rmse_std = np.std(model_final_rmse)
        avg_running_dev_rmse.append(sum(model_final_rmse_dev)/len(scores))
        avg_running_dev_rmse_std = np.std(model_final_rmse_dev)
        avg_running_train_rmse.append(sum(model_final_rmse_train)/len(scores))
        avg_running_train_rmse_std = np.std(model_final_rmse_train)


        if other_params[0]:
            logger.info('#################################')
            logger.info(f'-----Per model: {scores}-----')
            logger.info(f'average model rmse: {sum(model_final_rmse)/len(scores)}')
            logger.info(f'average model rmse std: {avg_running_rmse_std}')
            logger.info(f'--------------------------------')
            logger.info(f'average model train: {sum(model_final_rmse_train)/len(scores)}')
            logger.info(f'average model train std: {avg_running_dev_rmse_std}')
            logger.info(f'--------------------------------')
            logger.info(f'average model dev: {sum(model_final_rmse_dev)/len(scores)}')
            logger.info(f'average model dev std: {avg_running_train_rmse_std}')
            logger.info(f'--------------------------------')
            logger.info(f'median model alpha: {median_alpha}')
            logger.info('#################################')

        # -----------------------------------------------------
        # BETA = [reg_w, reg_h, reg_x, reg_y, reg_z, reg_bias_s, reg_bias_t]
        run_alpha_vec.append(median_alpha)

    # -----------------------------------------
    # find median hyperparam setting aver all runs
    # -----------------------------------------
    run_median_alpha = get_median_values(run_alpha_vec)

    running_rmse = sum(avg_running_rmse)/num_running
    running_rmse_std = np.std(avg_running_rmse)
    running_dev_rmse = sum(avg_running_dev_rmse)/num_running
    running_dev_rmse_std = np.std(avg_running_dev_rmse)
    running_train_rmse = sum(avg_running_train_rmse)/num_running
    running_train_rmse_std = np.std(avg_running_train_rmse)

    final_result = [running_rmse, running_dev_rmse, running_train_rmse, run_median_alpha]

    if other_params[0]:
        logger.info('#################################')
        logger.info(f'final_rmse run: {running_rmse}')
        logger.info(f'final_rmse run std: {running_rmse_std}')
        logger.info(f'--------------------------------')
        logger.info(f'final_rmse_train run: {running_train_rmse}')
        logger.info(f'final_rmse_train run std: {running_train_rmse_std}')
        logger.info(f'--------------------------------')
        logger.info(f'final_rmse_dev run : {running_dev_rmse}')
        logger.info(f'final_rmse_dev run : {running_dev_rmse_std}')
        logger.info(f'--------------------------------')
        logger.info(f'median_alpha run: {run_median_alpha}')
        logger.info('#################################')

        # -----------------------------------------------------
        # ------Create pickle file to dump all results---------
        # -----------------------------------------------------
        if other_params[12]:
            with open(path + '_' + filename + '.pkl', 'wb') as g:
                pkl.dump(results_all, g)

    return final_result


if __name__ == '__main__':

    debug = 1
    params = parse_args(debug=debug)

    if debug:
        print('--------------'*3)
        print("-------------------------DEBUGGING-------------------")
        print('--------------'*3)
    else:
        print('*******************' * 3)
        print("***********************Start Code************************")
        print('*******************' * 3)

    random.seed(276)
    np.random.seed(276)
    task = params.task
    attribute = params.attribute
    ctx = params.context
    STD = params.std  # Amount of noise to use for model initialization
    ALPHA = params.alpha  # fixed precision for the likelihood
    FOLDS = params.folds  # fold CV
    num_running = params.runs  # outer runs (NLPerf uses 10)
    DIM = params.dim
    split = params.split
    DIR = get_data(task, split)
    model = params.model

    SRC, TGT, SCORE, SIDE_FEATURES, CONTEXT_FEATURES, tgt_si_len, src_si_len, lang_pair_si_len, \
    SIDE_INFO_DICT, SRC_SIDE_INFO_DICT, TGT_SIDE_INFO_DICT, ctx, attribute = get_info(task, attribute, ctx, params, DIR[1])
    filename = get_filename(params, ctx)
    host_name = socket.gethostname()
    eval_on_dev = 1
    path, pathlogs = create_directory(task, host_name, params.savepngs, attribute, FOLDS, num_running, ctx, DIM, model)

    other_params = [params.enable_log, params.doplots, params.savepngs, split, task, attribute, ctx, eval_on_dev,
                    params.chains, params.tune, params.draws, params.burn, params.uncertainty_analysis]

    if other_params[0]:
        if task == 'bli_aaMF':
            ctx = 'Uriel distance features'
        if attribute == "si":
            ctx = None
        logger = create_logger(f"{pathlogs}/Bayes_{filename}.log")
        logger.info('#' * 100)
        logger.info(f'Start: Task: {task} | model: {model} | attribute: {attribute} | context: {ctx} | debug: {debug} |')
        logger.info('#' * 100)
    else:
        logger = None

    result = get_result(filename, path, logger, other_params, DIM, ALPHA, DIR, SCORE, SRC, TGT, FOLDS, num_running, STD,\
                             src_lang_side_info=SRC_SIDE_INFO_DICT, \
                             tgt_lang_side_info=TGT_SIDE_INFO_DICT, \
                             lang_pair_side_info=SIDE_INFO_DICT, \
                             src_si_len=src_si_len, tgt_si_len=tgt_si_len, lang_pair_si_len=lang_pair_si_len)