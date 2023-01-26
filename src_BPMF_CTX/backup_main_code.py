import numpy as np
import random
from utils import parse_args
import socket
from mf import MF
from helpers import generate_score_matrix, get_rmse, get_permutations, get_median_values, create_directory
from get_information import get_info, get_context, get_filename
from split_functions import split_k_fold_data, split_k_fold_data_ncv
from logger import create_logger


def get_result(logger, other_params, dim, alpha, beta, data_dir, scores, src_index, tgt_index, k, num_running, iterations, src_lang_side_info=None,
               tgt_lang_side_info=None, lang_pair_side_info=None, src_si_len=0, tgt_si_len=0,
               lang_pair_si_len=0):

    hyperparam_settings = get_permutations(alpha, beta)

    if other_params[0]:
        logger.info(f'hyperparameters:')
        logger.info(f'ALPHA (lr) = {alpha}, BETA = [reg_w = {beta[0]}, reg_h = {beta[1]}, reg_x = {beta[2]}, reg_y = {beta[3]}, reg_z = {beta[4]}, reg_bias_s = {beta[5]}, reg_bias_t = {beta[6]}]')
        logger.info(f'leading to... {len(hyperparam_settings)} different hyperparameter settings')
        logger.info('*'*20)

    avg_running_rmse, avg_running_dev_rmse, avg_running_train_rmse = 0.0, 0.0, 0.0

    # Record best hyperparam settings for each run
    # -----------------------------------------------------
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
        if other_params[3] == 'NCV':
            data, langs_matrix = split_k_fold_data_ncv(data_dir, scores, src_index, tgt_index, k)

        for model in scores:

            if other_params[3] == 'NCV':
                if other_params[0]:
                    logger.info('*' * 20)
                    logger.info(f'Scores: {model} using NCV')
                    logger.info('*' * 20)

                res, fold = {}, {}
                average_rmse, average_rmse_train, average_rmse_dev = 0.0, 0.0, 0.0

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
                    train_data_full, test_data = data[model]["train"][i], data[model]["test"][i]

                    # --------------------------------------------------
                    # A loop to optimize over hyperparameters
                    # --------------------------------------------------
                    for hypset in range(len(hyperparam_settings)):

                        alpha = hyperparam_settings[hypset][hypset+1]['alpha']
                        beta = hyperparam_settings[hypset][hypset+1]['beta']

                        if other_params[0]:
                            logger.info('*' * 20)
                            logger.info(f'hyperparam setting: {hypset+1}')
                            logger.info(f'hyperparameters:')
                            logger.info(f'ALPHA (lr) = {alpha}, BETA = [reg_w = {beta[0]}, reg_h = {beta[1]}, reg_x = {beta[2]}, reg_y = {beta[3]}, reg_z = {beta[4]}, reg_bias_s = {beta[5]}, reg_bias_t = {beta[6]}]')
                            logger.info('*' * 20)

                        total_train_rmse_cv, total_dev_rmse_cv, total_rmse_cv, \
                        argmin_h_dev_error = 0.0, 0.0, 0.0, {'average_dev': 1000}

                        # --------------------------------------------------
                        # Inner CV loop
                        # --------------------------------------------------
                        for ii in range(k-1):
                            train_data, dev_data = train_data_full["train"][ii], train_data_full["dev"][ii]

                            train_matrix = generate_score_matrix(train_data, src_index, tgt_index, model, langs_matrix)

                            mf = MF(train_matrix, other_params, K=dim, alpha=alpha, beta=beta, iterations=iterations, dev_samples=dev_data,
                                    X=src_lang_side_info, Y=tgt_lang_side_info, Z=lang_pair_side_info, src_si_len=src_si_len, tgt_si_len=tgt_si_len,
                                    lang_pair_si_len=lang_pair_si_len, src_index=src_index, tgt_index=tgt_index, model=model,
                                    num_running=nr)

                            # Get train, dev RMSEs
                            # ------------------------------
                            training_log = mf.train()
                            cur_rmse_train = training_log[0][-1][1]
                            cur_rmse_dev = training_log[1][-1][1]

                            # Get test, dev RMSEs
                            # ------------------------------
                            cur_rmse = mf.evaluate_testing(test_data, src_index, tgt_index, model)

                            if other_params[1]:
                                mf.draw_error_curve(ii, pngpaths)

                            total_train_rmse_cv += cur_rmse_train
                            total_dev_rmse_cv += cur_rmse_dev
                            total_rmse_cv += cur_rmse

                            if other_params[0]:
                                logger.info("\t\t(cv fold {}) train RMSE is {}.".format(ii+1, cur_rmse_train))
                                logger.info("\t\t(cv fold {}) dev RMSE is {}.".format(ii+1, cur_rmse_dev))
                                logger.info("\t\t(cv fold {}) test RMSE is {}.".format(ii+1, cur_rmse))
                                logger.info("*" * 20)

                        average_dev = total_dev_rmse_cv / (k-1)
                        average_train = total_train_rmse_cv / (k - 1)
                        average_test = total_rmse_cv / (k - 1)

                        if argmin_h_dev_error['average_dev'] > average_dev:
                            argmin_h_dev_error = {'average_dev': average_dev, \
                                                  'average_test': average_test,\
                                                  'average_train': average_train,
                                                  'alpha': alpha, 'beta': beta}

                    if other_params[0]:
                        logger.info(f"(Fold {i}) min average dev rmse: " + str(argmin_h_dev_error['average_dev']))
                    fold[f'fold {i}'] = argmin_h_dev_error
                    res[model] = fold

                    average_rmse += argmin_h_dev_error['average_test']
                    average_rmse_train += argmin_h_dev_error['average_train']
                    average_rmse_dev += argmin_h_dev_error['average_dev']
                    alpha_vec.append(argmin_h_dev_error['alpha']), reg_w_vec.append(argmin_h_dev_error['beta'][0]), reg_h_vec.append(argmin_h_dev_error['beta'][1])
                    reg_x_vec.append(argmin_h_dev_error['beta'][2]), reg_y_vec.append(argmin_h_dev_error['beta'][3]), reg_z_vec.append(argmin_h_dev_error['beta'][4])
                    reg_bias_s_vec.append(argmin_h_dev_error['beta'][5]), reg_bias_t_vec.append(argmin_h_dev_error['beta'][6])

                # find average error here
                # -----------------------------------------
                final_rmse = average_rmse / k
                final_rmse_train = average_rmse_train / k
                final_rmse_dev = average_rmse_dev / k

                # -----------------------------------------
                # find median hyperparam setting
                # -----------------------------------------
                median_alpha, median_reg_w, median_reg_h, median_reg_bias_s, median_reg_bias_t, \
                median_reg_x, median_reg_y, median_reg_z = get_median_values(alpha_vec, reg_w_vec, reg_h_vec, \
                                                    reg_bias_s_vec, reg_bias_t_vec, reg_x_vec, reg_y_vec, reg_z_vec)

                if other_params[0]:
                    logger.info('#################################')
                    logger.info(f'final_rmse: {final_rmse}')
                    logger.info(f'final_rmse_train: {final_rmse_train}')
                    logger.info(f'final_rmse_dev: {final_rmse_dev}')
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

        # -----------------------------------------
        # Average per run here
        # -----------------------------------------

        avg_running_rmse += final_rmse
        avg_running_dev_rmse += final_rmse_dev
        avg_running_train_rmse += final_rmse_train

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
    running_rmse = avg_running_rmse/num_running
    running_dev_rmse = avg_running_dev_rmse/num_running
    running_train_rmse = avg_running_train_rmse/num_running

    final_result = [running_rmse, running_dev_rmse, running_train_rmse, run_median_alpha,\
        [run_median_reg_w, run_median_reg_h,run_median_reg_x, run_median_reg_y,\
         run_median_reg_z, run_median_reg_bias_s, run_median_reg_bias_t]]

    if other_params[0]:
        logger.info('#################################')
        logger.info(f'final_rmse run: {running_rmse}')
        logger.info(f'final_rmse_train run: {running_train_rmse}')
        logger.info(f'final_rmse_dev run : {running_dev_rmse}')
        logger.info(f'median_alpha run: {run_median_alpha}')
        logger.info(f'median_beta run: {[run_median_reg_w, run_median_reg_h,run_median_reg_x, run_median_reg_y, run_median_reg_z, run_median_reg_bias_s, run_median_reg_bias_t]}')
        logger.info('#################################')

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
    ALPHA = params.lr  # learning rate
    FOLDS = params.folds  # fold CV
    num_running = params.runs  # outer runs (NLPerf uses 10)
    iterations = params.iterations
    DIM = params.dim
    DIR = f"../data/data_{task}.csv"

    SRC, TGT, SCORE, SIDE_FEATURES, CONTEXT_FEATURES, tgt_si_len, src_si_len, lang_pair_si_len, \
    BETA, SIDE_INFO_DICT, SRC_SIDE_INFO_DICT, TGT_SIDE_INFO_DICT, ctx = get_info(task, attribute, ctx, params, DIR)
    filename = get_filename(params, ctx)
    host_name = socket.gethostname()

    path = create_directory(task, host_name)

    other_params = [params.enable_log, params.doplots, params.savepngs, params.split, task, attribute, ctx, params.debug_reg_z]

    if other_params[0]:
        logger = create_logger(f"{path}/{filename}.log")
        logger.info('#' * 100)
        logger.info(f'Start: Task: {task} | attribute: {attribute} | context: {ctx} | debug: {debug} | debug reg_z: {params.debug_reg_z}')
        logger.info('#' * 100)
    else:
        logger = None

    result = get_result(logger, other_params, DIM, ALPHA, BETA, DIR, SCORE, SRC, TGT, FOLDS, num_running, iterations, \
                             src_lang_side_info=SRC_SIDE_INFO_DICT, \
                             tgt_lang_side_info=TGT_SIDE_INFO_DICT, \
                             lang_pair_side_info=SIDE_INFO_DICT, \
                             src_si_len=src_si_len, tgt_si_len=tgt_si_len, lang_pair_si_len=lang_pair_si_len)