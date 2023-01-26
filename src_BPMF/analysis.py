# Code taken from https://docs.pymc.io/en/stable/pymc-examples/examples/case_studies/probabilistic_matrix_factorization.html

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Make tex plots
# plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.style.use("seaborn-darkgrid")


def analyse_map(baselines, mf_map_rmse, fold, run):
    mf_improvement_mom = baselines["mom"] - mf_map_rmse
    mf_improvement_ur = baselines["ur"] - mf_map_rmse
    mf_improvement_gm = baselines["gm"] - mf_map_rmse
    print('-' * 20)
    print(f'fold: {fold}, run: {run}')
    print("MF MAP Improvement to UniformRandom Baseline:   %.5f" % mf_improvement_ur)
    print("MF MAP Improvement to GlobalMeans Baseline:   %.5f" % mf_improvement_gm)
    print("MF MAP Improvement to MeanOfMeans Baseline:   %.5f" % mf_improvement_mom)


def analyse_mcmc(mf, _analysis, path, draws, tune, chains, DIM, ALPHA, fold, run, model):

    fig = mf.traceplot(draws, tune, chains, DIM, ALPHA, fold, run, model)
    if _analysis['save']:
        if _analysis['png_save']:
            fig.savefig(path + '_traceplot.png', bbox_inches='tight')
        if _analysis['latex_save']:
            fig.savefig(path + '_traceplot.eps', bbox_inches='tight')
        plt.close('all')


def analyse_rmse_mcmc(logger, other_params, mf, train, test, fold, run, i, path, alpha, flag_test=0):
    predicted, results = mf.running_rmse(np.array(test), np.array(train), run, i, fold, path, other_params[2], other_params[1], alpha, flag_test = flag_test, burn_in=other_params[11])

    # -----------------------
    # Dev RMSE
    # -----------------------
    if flag_test == 0:
        dev_rmse = results["running-dev"].values[-1]
        train_rmse = results["running-train"].values[-1]
        if other_params[0]:
            logger.info('-' * 20)
            logger.info(f'Outer CV fold: {i+1} | Inner CV fold: {fold+1} | run: {run+1}')
            logger.info('-' * 20)
            logger.info("Posterior predictive train RMSE: %.5f" % train_rmse)
            logger.info("Posterior predictive dev RMSE:  %.5f" % dev_rmse)
            logger.info("Train/dev difference:           %.5f" % (dev_rmse - train_rmse))
            logger.info('-' * 20)
    else:
        test_rmse = results["running-test"].values[-1]
        train_rmse = results["running-train"].values[-1]
        if other_params[0]:
            logger.info('-' * 20)
            logger.info(f'Outer CV fold: {i+1} || run: {run+1}')
            logger.info('-' * 20)
            logger.info("Posterior predictive train RMSE: %.5f" % train_rmse)
            logger.info("Posterior predictive test RMSE:  %.5f" % test_rmse)
            logger.info("Train/test difference:           %.5f" % (test_rmse - train_rmse))
            logger.info('-' * 20)

    return predicted, results, test_rmse, train_rmse


def analyse_rmse(logistic_model, burn, mf, mf_map_rmse, train, test, fold, run):
    
    predicted, results = mf.running_rmse(logistic_model, test, train, burn_in=burn)
    # -----------------------
    # Final RMSE
    # -----------------------
    final_test_rmse = results["running-test"].values[-1]
    final_train_rmse = results["running-train"].values[-1]
    print('-' * 20)
    print(f'fold: {fold+1}, run: {run+1}')
    print("Posterior predictive train RMSE: %.5f" % final_train_rmse)
    print("Posterior predictive test RMSE:  %.5f" % final_test_rmse)
    print("Train/test difference:           %.5f" % (final_test_rmse - final_train_rmse))
    print("Improvement from MAP:            %.5f" % (mf_map_rmse - final_test_rmse))
    
    return predicted, results, final_test_rmse, final_train_rmse


def analyse_summary(mf_map_rmse, results, fold, run, _analysis,
                    path, draws, tune, chains, DIM, ALPHA, model):
    size = 100  # RMSE doesn't really change after 100th sample anyway.
    all_results = pd.DataFrame(
        {
            "PMF MAP": np.repeat(mf_map_rmse, size),
            "PMF MCMC": results["running-test"][:size],
        }
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    all_results.plot(kind="line", grid=False, ax=ax, title=f"")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("RMSE")
    plt.suptitle(
        f'RMSE for all methods \n Run: {run}, Fold: {fold}, Model: {model}, Dim: {DIM}, \n'
        f' Alpha: {ALPHA}, Chains: {chains}, Draws: {draws}, Tune: {tune}')
    
    if _analysis['save']:
        if _analysis['png_save']:
            fig.savefig(path + '_summary.png', bbox_inches='tight')
        if _analysis['latex_save']:
            fig.savefig(path + '_summary.eps', bbox_inches='tight')
        plt.close('all')


