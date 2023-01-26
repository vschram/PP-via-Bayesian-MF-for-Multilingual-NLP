##################################################
### Code based on ThianziLi and NLPerf
#################################################

import numpy as np
import random
from utils import parse_args
import socket
from mf import MF
from helpers import generate_score_matrix, get_rmse, get_permutations, get_median_values, create_directory, get_data
from get_information import get_info, get_context, get_filename
from split_functions import split_k_fold_data, split_k_fold_data_ncv, split_k_fold_data_nlperf, split_k_fold_data_lolo_source, split_k_fold_data_lolo_target
from logger import create_logger
from copy import deepcopy


def get_result(path, logger, other_params, dim, alpha, beta, data_dir, scores, src_index, tgt_index, k, num_running, iterations, src_lang_side_info=None,
               tgt_lang_side_info=None, lang_pair_side_info=None, src_si_len=0, tgt_si_len=0,
               lang_pair_si_len=0):

    hyperparam_settings = get_permutations(alpha, beta)

    if other_params[0]:
        logger.info(f'hyperparameters:')
        logger.info(f'ALPHA (lr) = {alpha}, BETA = [reg_w = {beta[0]}, reg_h = {beta[1]}, reg_x = {beta[2]}, reg_y = {beta[3]}, reg_z = {beta[4]}, reg_bias_s = {beta[5]}, reg_bias_t = {beta[6]}]')
        logger.info(f'leading to... {len(hyperparam_settings)} different hyperparameter settings')
        logger.info('*'*20)


    # Record best hyperparam settings for each run
    # -----------------------------------------------------
    avg_running_rmse, avg_running_dev_rmse, avg_running_train_rmse = [], [], []
    # BETA = [reg_w, reg_h, reg_x, reg_y, reg_z, reg_bias_s, reg_bias_t]
    run_alpha_vec, run_reg_w_vec, run_reg_h_vec, run_reg_x_vec, run_reg_y_vec, run_reg_z_vec, \
    run_reg_bias_s_vec, run_reg_bias_t_vec = [], [], [], [], [], [], [], []

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

        if 'lolo_source' in split:
            k = langs_matrix.shape[0]
        elif 'lolo_target' in split:
            k = langs_matrix.shape[1]

        model_final_rmse, model_final_rmse_dev, model_final_rmse_train = [], [], []
        # BETA = [reg_w, reg_h, reg_x, reg_y, reg_z, reg_bias_s, reg_bias_t]
        model_alpha_vec, model_reg_w_vec, model_reg_h_vec, model_reg_x_vec, model_reg_y_vec, model_reg_z_vec, \
        model_reg_bias_s_vec, model_reg_bias_t_vec = [], [], [], [], [], [], [], []

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
                # BETA = [reg_w, reg_h, reg_x, reg_y, reg_z, reg_bias_s, reg_bias_t]
                alpha_vec, reg_w_vec, reg_h_vec, reg_x_vec, reg_y_vec, reg_z_vec, reg_bias_s_vec, reg_bias_t_vec \
                    = [], [], [], [], [], [], [], []

                # Start k-fold
                # -----------------------------------------------------
                for i in range(k):
                    if other_params[0]:
                        logger.info("\tNCV, Fold {}: ".format(i + 1))

                    # Create test set here, but use in every inner CV too, so later no retraining is required.
                    train_data_full, test_data, train_data_complete = data[model]["train"][i], data[model]["test"][i], data[model]["full_train"][i]




                    if len(hyperparam_settings)==1:

                        argmin_h_dev_error = {'average_dev': 0, 'alpha': hyperparam_settings[0][1]['alpha'], 'beta': hyperparam_settings[0][1]['beta'], 'h': 1}

                        if other_params[0]:
                            logger.info('*' * 20)
                            logger.info('Only one hyperparam setting, no optimization needed')
                            logger.info('*' * 20)

                    else:
                        # --------------------------------------------------
                        # A loop to optimize over hyperparameters
                        # --------------------------------------------------
                        argmin_h_dev_error = {'average_dev': 1000, 'h': 1000, 'alpha': 0.1, 'beta': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]}

                        for hypset in range(len(hyperparam_settings)):
                            alpha = hyperparam_settings[hypset][hypset+1]['alpha']
                            beta = hyperparam_settings[hypset][hypset+1]['beta']

                            total_train_rmse_cv, total_dev_rmse_cv, total_rmse_cv = 0.0, 0.0, 0.0

                            if other_params[0]:
                                logger.info('*' * 20)
                                logger.info(f'hyperparam setting: {hypset+1}')
                                logger.info(f'hyperparameters:')
                                logger.info(f'ALPHA (lr) = {alpha}, BETA = [reg_w = {beta[0]}, reg_h = {beta[1]}, reg_x = {beta[2]}, reg_y = {beta[3]}, reg_z = {beta[4]}, reg_bias_s = {beta[5]}, reg_bias_t = {beta[6]}]')
                                logger.info('*' * 20)

                            # --------------------------------------------------
                            # Inner CV loop
                            # --------------------------------------------------
                            for ii in range(k-1):
                                train_data, dev_data = train_data_full["train"][ii], train_data_full["dev"][ii]

                                train_matrix = generate_score_matrix(train_data, src_index, tgt_index, model, langs_matrix)

                                mf = MF(logger, train_matrix, other_params, K=dim, alpha=alpha, beta=beta, iterations=iterations, dev_samples=dev_data,
                                        X=src_lang_side_info, Y=tgt_lang_side_info, Z=lang_pair_side_info, src_si_len=src_si_len, tgt_si_len=tgt_si_len,
                                        lang_pair_si_len=lang_pair_si_len, src_index=src_index, tgt_index=tgt_index, model=model,
                                        num_running=nr)

                                # Get train, dev RMSEs
                                # ------------------------------
                                training_log = mf.train()
                                cur_rmse_dev = training_log[1][-1][1]

                                if other_params[1]:
                                    mf.draw_error_curve(nr, i, ii, path, other_params[2], other_params[1], alpha, beta)

                                total_dev_rmse_cv += cur_rmse_dev

                                if other_params[0]:
                                    logger.info('*' * 20)
                                    logger.info("\t\t(Inner cv fold {}) dev RMSE is {}.".format(ii+1, cur_rmse_dev))
                                    logger.info("*" * 20)

                            average_dev = total_dev_rmse_cv / (k-1)

                            if argmin_h_dev_error['average_dev'] > average_dev:
                                argmin_h_dev_error = {'average_dev': average_dev, 'alpha': alpha, 'beta': beta, 'h': hypset+1}

                        if other_params[0]:
                            logger.info("#" * 20)
                            logger.info(f"(Outer fold {i+1}) min average dev rmse, averaged over {ii+1} inner folds: "
                                        + str(argmin_h_dev_error['average_dev'])
                                        + str(argmin_h_dev_error['h']))
                            logger.info("#" * 20)

                    # ----------------------------------------------------------
                    # -- Train model on complete train set with h that causes min dev error
                    # ----------------------------------------------------------
                    other_params[8] = 0
                    train_matrix_full = generate_score_matrix(train_data_complete, src_index, tgt_index, model,
                                                              langs_matrix)
                    mf_full = MF(logger, train_matrix_full, other_params, K=dim, alpha=argmin_h_dev_error['alpha'], beta=argmin_h_dev_error['beta'],
                                 iterations=iterations, dev_samples=None,
                                 X=src_lang_side_info, Y=tgt_lang_side_info, Z=lang_pair_side_info,
                                 src_si_len=src_si_len, tgt_si_len=tgt_si_len,
                                 lang_pair_si_len=lang_pair_si_len, src_index=src_index, tgt_index=tgt_index,
                                 model=model,
                                 num_running=nr)
                    training_log = mf_full.train()
                    cur_rmse_train = training_log[0][-1][1]
                    cur_rmse = mf_full.evaluate_testing(test_data, src_index, tgt_index, model)

                    if other_params[0]:
                        logger.info('*' * 20)
                        logger.info("\t\t(Outer Fold {}) train RMSE is {}.".format(i + 1, cur_rmse_train))
                        logger.info("\t\t(Outer Fold {}) test RMSE is {}.".format(i + 1, cur_rmse))
                        logger.info(argmin_h_dev_error)
                        logger.info("*" * 20)
                    other_params[8] = 1
                    # ----------------------------------------------------------
                    # ----------------------------------------------------------
                    # ----------------------------------------------------------

                    #fold[f'fold {i}'] = argmin_h_dev_error
                    #res[model] = fold
                    average_rmse.append(cur_rmse)
                    average_rmse_train.append(cur_rmse_train)
                    average_rmse_dev.append(argmin_h_dev_error['average_dev'])
                    alpha_vec.append(argmin_h_dev_error['alpha']), reg_w_vec.append(argmin_h_dev_error['beta'][0]), reg_h_vec.append(argmin_h_dev_error['beta'][1])
                    reg_x_vec.append(argmin_h_dev_error['beta'][2]), reg_y_vec.append(argmin_h_dev_error['beta'][3]), reg_z_vec.append(argmin_h_dev_error['beta'][4])
                    reg_bias_s_vec.append(argmin_h_dev_error['beta'][5]), reg_bias_t_vec.append(argmin_h_dev_error['beta'][6])


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
                median_alpha, median_reg_w, median_reg_h, median_reg_bias_s, median_reg_bias_t, \
                median_reg_x, median_reg_y, median_reg_z = get_median_values(alpha_vec, reg_w_vec, reg_h_vec, \
                                                    reg_bias_s_vec, reg_bias_t_vec, reg_x_vec, reg_y_vec, reg_z_vec)

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
                total_rmse = 0.0
                res = {}
                for i in range(k):
                    print("\tFold {}: ".format(i + 1))
                    train_data, test_data = data[model]["train"][i], data[model]["test"][i]
                    train_matrix = generate_score_matrix(train_data, src_index, tgt_index, model, langs_matrix)
                    mf = MF(train_matrix, other_params, K=dim, alpha=alpha, beta=beta, iterations=iterations, dev_samples=test_data,
                            X=src_lang_side_info,
                            Y=tgt_lang_side_info, Z=lang_pair_side_info, src_si_len=src_si_len, tgt_si_len=tgt_si_len,
                            lang_pair_si_len=lang_pair_si_len, src_index=src_index, tgt_index=tgt_index, model=model,
                            num_running=nr)
                    training_log = mf.train()
                    #                 predictions = mf.full_matrix()
                    #                 cur_rmse = get_rmse(test_data, model, src_index, tgt_index, predictions, train_matrix)
                    cur_rmse = mf.evaluate_testing(test_data, src_index, tgt_index, model)
                    if other_params[1]:
                        mf.draw_error_curve(i)
                    total_rmse += cur_rmse
                    print("\t\trmse is {}.".format(cur_rmse))
                    print("*" * 20)

                average_rmse = total_rmse / k
                print("average rmse: " + str(average_rmse))
                res[model] = average_rmse

            # -----------------------------------------
            # Average per model here
            # -----------------------------------------
            model_final_rmse.append(final_rmse)
            model_final_rmse_train.append(final_rmse_train)
            model_final_rmse_dev.append(final_rmse_dev)

            # BETA = [reg_w, reg_h, reg_x, reg_y, reg_z, reg_bias_s, reg_bias_t]
            model_alpha_vec.append(median_alpha), model_reg_w_vec.append(median_reg_w), model_reg_h_vec.append(median_reg_h)
            model_reg_x_vec.append(median_reg_x), model_reg_y_vec.append(median_reg_y), model_reg_z_vec.append(median_reg_z)
            model_reg_bias_s_vec.append(median_reg_bias_s), model_reg_bias_t_vec.append(median_reg_bias_t)


        # -----------------------------------------
        # find median hyperparam setting aver all models
        # -----------------------------------------
        median_alpha, median_reg_w, median_reg_h, median_reg_bias_s, median_reg_bias_t, \
        median_reg_x, median_reg_y, median_reg_z = get_median_values(model_alpha_vec, model_reg_w_vec,
                                                                                 model_reg_h_vec, \
                                                                                 model_reg_bias_s_vec, model_reg_bias_t_vec,
                                                                                 model_reg_x_vec, model_reg_y_vec,
                                                                                 model_reg_z_vec)
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
        run_alpha_vec.append(median_alpha), run_reg_w_vec.append(median_reg_w), run_reg_h_vec.append(median_reg_h)
        run_reg_x_vec.append(median_reg_x), run_reg_y_vec.append(median_reg_y), run_reg_z_vec.append(median_reg_z)
        run_reg_bias_s_vec.append(median_reg_bias_s), run_reg_bias_t_vec.append(median_reg_bias_t)



    # -----------------------------------------
    # find median hyperparam setting aver all runs
    # -----------------------------------------
    run_median_alpha, run_median_reg_w, run_median_reg_h, run_median_reg_bias_s, run_median_reg_bias_t, \
    run_median_reg_x, run_median_reg_y, run_median_reg_z = get_median_values(run_alpha_vec, run_reg_w_vec, run_reg_h_vec, \
                                                                 run_reg_bias_s_vec, run_reg_bias_t_vec, run_reg_x_vec, run_reg_y_vec,
                                                                 run_reg_z_vec)

    running_rmse = sum(avg_running_rmse)/num_running
    running_rmse_std = np.std(avg_running_rmse)
    running_dev_rmse = sum(avg_running_dev_rmse)/num_running
    running_dev_rmse_std = np.std(avg_running_dev_rmse)
    running_train_rmse = sum(avg_running_train_rmse)/num_running
    running_train_rmse_std = np.std(avg_running_train_rmse)

    final_result = [running_rmse, running_dev_rmse, running_train_rmse, run_median_alpha, \
        [run_median_reg_w, run_median_reg_h, run_median_reg_x, run_median_reg_y, \
         run_median_reg_z, run_median_reg_bias_s, run_median_reg_bias_t]]

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
        logger.info(f'median_beta run: {[run_median_reg_w, run_median_reg_h,run_median_reg_x, run_median_reg_y, run_median_reg_z, run_median_reg_bias_s, run_median_reg_bias_t]}')
        logger.info('#################################')

    return final_result


if __name__ == '__main__':

    debug = 0
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
    ALPHA = params.lr  # learning rate
    FOLDS = params.folds  # fold CV
    num_running = params.runs  # outer runs (NLPerf uses 10)
    iterations = params.iterations
    DIM = params.dim
    split = params.split
    DIR = get_data(task, split)

    SRC, TGT, SCORE, SIDE_FEATURES, CONTEXT_FEATURES, tgt_si_len, src_si_len, lang_pair_si_len, \
    BETA, SIDE_INFO_DICT, SRC_SIDE_INFO_DICT, TGT_SIDE_INFO_DICT, ctx, attribute = get_info(task, attribute, ctx, params, DIR[1])
    filename = get_filename(params, ctx)
    host_name = socket.gethostname()
    eval_on_dev = 1
    path, pathlogs = create_directory(task, host_name, params.savepngs, attribute, FOLDS, num_running, ctx, DIM)

    other_params = [params.enable_log, params.doplots, params.savepngs, split, task, attribute, ctx,
                    params.debug_reg_z, eval_on_dev, params.diff_reg, params.lr_decay]

    if other_params[0]:
        if task == 'bli_aaMF':
            ctx = 'Uriel distance features'
        if attribute == "si":
            ctx = None
        logger = create_logger(f"{pathlogs}/{filename}.log")
        logger.info('#' * 100)
        logger.info(f'Start: Task: {task} | attribute: {attribute} | context: {ctx} | debug: {debug} | debug reg_z: {params.debug_reg_z}')
        logger.info('#' * 100)
    else:
        logger = None

    result = get_result(path, logger, other_params, DIM, ALPHA, BETA, DIR, SCORE, SRC, TGT, FOLDS, num_running, iterations, \
                             src_lang_side_info=SRC_SIDE_INFO_DICT, \
                             tgt_lang_side_info=TGT_SIDE_INFO_DICT, \
                             lang_pair_side_info=SIDE_INFO_DICT, \
                             src_si_len=src_si_len, tgt_si_len=tgt_si_len, lang_pair_si_len=lang_pair_si_len)


def generate_score_matrix(train_data, src_index_name, tgt_index_name, score_index_name, origin_score_matrix):
    score_matrix = deepcopy(origin_score_matrix)

    for record in train_data.iterrows():
        record = record[1]
        src_lang = record[src_index_name]
        tgt_lang = record[tgt_index_name]
        score = record[score_index_name]
        score_matrix.loc[src_lang, tgt_lang] = score

    return score_matrix