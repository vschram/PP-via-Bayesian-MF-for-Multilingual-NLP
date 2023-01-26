# Based on: https://docs.pymc.io/en/stable/pymc-examples/examples/case_studies/probabilistic_matrix_factorization.html

import pymc3 as pm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import arviz as az
# Enable on-the-fly graph computations, but ignore
# absence of intermediate test values.
# theano.config.compute_test_value = "raise"
import seaborn as sns

class PMF:
    """Probabilistic Matrix Factorization model using pymc3."""

    def __init__(self, mod, logger, other_params, data, var, dim=2, alpha=2, std=0.01):
        """Build the Probabilistic Matrix Factorization model using pymc3.
        :param np.ndarray data: The training data to use for learning the model.
        :param int dim: Dimensionality of the model; number of latent factors.
        :param int alpha: Fixed precision for the likelihood function.
        :param float std: Amount of noise to use for model initialization.
        """

        # --------------------------------------
        # get params
        #--------------------------------------
        # Score matrix
        self.R = np.array(data)
        self.score_dict = {'BLEU': "WIKI-MT", "Muse": "BLI-Muse", "Vecmap": "BLI-Vecmap"}
        self.score = mod
        self.attribute = other_params[5]
        self.context_number = other_params[6]
        self.var_w = var[0]
        self.var_h = var[1]
        self.dim = dim
        self.alpha = alpha
        # STD = sqrt(variance)
        self.std = np.sqrt(1.0 / alpha)
        self.data = self.R.copy()
        self.data_orig = self.R.copy()
        n, m = self.data.shape


        # Perform mean value imputation
        nan_mask = np.isnan(self.data)
        self.data[nan_mask] = self.data[~nan_mask].mean()

        # Low precision reflects uncertainty; prevents overfitting.
        # Set to the mean variance across users and items.
        self.alpha_w = 1 / self.var_w
        self.alpha_h = 1 / self.var_h

        self.other_params = other_params

        # Specify the model.
        if other_params[0]:
            logger.info('*' * 20)
            logger.info("building the PMF model")

        self.model = pm.Model()
        self.map = None

        with self.model as pmf:

            W = pm.MvNormal(
                "W",
                mu=0,
                tau=self.alpha_w * np.eye(dim),
                shape=(n, dim),
                testval=np.random.randn(n, dim) * std,
            )

            H = pm.MvNormal(
                "H",
                mu=0,
                tau=self.alpha_h * np.eye(dim),
                shape=(m, dim),
                testval=np.random.randn(m, dim) * std,
            )

            R = pm.Normal("R", mu=(W @ H.T)[~nan_mask], tau=self.alpha, observed=self.data[~nan_mask])

        # Build probabilistic graph - works only locally
        #gv = pm.model_to_graphviz(pmf)
        #gv.render(filename='pmf_model', format='pdf')

        if other_params[0]:
            logger.info('*' * 20)
            logger.info("done building the PMF model")
        self.model = pmf

    def __str__(self):
        return self.name

    # Draw MCMC samples.
    def draw_samples(self, **kwargs):
        # kwargs.setdefault("chains", 1)  # trace = pm.sample(1000, tune=1000)
        with self.model:
            self.trace = pm.sample(**kwargs)  #**kwargs,
        return self.trace

    def predict(self, W, H):
        """Estimate R from the given values of U and V."""
        R = W @ H.T
        n, m = R.shape
        sample_R = np.random.normal(R, self.std)
        return sample_R

    def running_rmse(self, test_data, train_data,  nr, i, ii, path, savepngs, doplots, alpha, beta, flag_test=0, burn_in=0):
        """Calculate RMSE for each step of the trace to monitor convergence."""
        burn_in = burn_in if len(self.trace) >= burn_in else 0

        if flag_test == 0:
            results = {"error-per_cell": [], "running-mean-samples": [], "running-var-samples": [], "per-step-train": [], "running-train": [], "per-step-dev": [], "running-dev": []}
        else:
            results = {"error-per_cell": [], "running-mean-samples": [], "running-var-samples": [], "per-step-train": [], "running-train": [],
                       "per-step-test": [], "running-test": []}

        R = np.zeros(test_data.shape)
        R_samples = []
        for cnt, sample in enumerate(self.trace[burn_in:]):
            sample_R = self.predict(sample["W"], sample["H"])
            R += sample_R
            R_samples.append(sample_R)
            running_R = R / (cnt + 1)  # This is the mean of the samples
            running_var_R = np.var(R_samples, axis=0)
            running_mean_R = np.mean(R_samples, axis=0)  # This too. Redundant.
            if flag_test == 0:
                results["error-per_cell"].append(train_data - sample_R)
                results["per-step-train"].append(self.rmse(train_data, sample_R))
                results["running-train"].append(self.rmse(train_data, running_R))
                results["per-step-dev"].append(self.rmse(test_data, sample_R))
                results["running-dev"].append(self.rmse(test_data, running_R))
                results["running-var-samples"].append(running_var_R)
                results["running-mean-samples"].append(running_mean_R)
            else:
                results["error-per_cell"].append(train_data - sample_R)
                results["per-step-train"].append(self.rmse(train_data, sample_R))
                results["running-train"].append(self.rmse(train_data, running_R))
                results["per-step-test"].append(self.rmse(test_data, sample_R))
                results["running-test"].append(self.rmse(test_data, running_R))
                results["running-var-samples"].append(running_var_R)
                results["running-mean-samples"].append(running_mean_R)

        results = pd.DataFrame(results)



        if doplots:
            results.plot(
                kind="line",
                grid=True,
                figsize=(15, 7),
                xlabel='Draws',
                ylabel='Value',
                title=f'| Posterior Predictive |  | Run: {nr} | Fold: {i} | CV fold: {ii}|'
            )

            if savepngs:
                pngpath = f"{path}/dim_{self.dim}/attribute_{self.attribute}/ctx_number_{self.context_number}/run_{nr+1}/fold_{i+1}"
                filename = f'{alpha}_{beta[0]}_{beta[1]}_{beta[2]}_{beta[3]}_{beta[4]}'
                plt.savefig(f"{pngpath}" + "/" + filename + ".png")
            # plt.show()
            plt.close()

            with self.model:
                ppc = pm.sample_posterior_predictive(self.trace, var_names=["W", "H", "R"], random_seed=2706)
                az.plot_ppc(az.from_pymc3(posterior_predictive=ppc, model=self.model))
                if savepngs:
                    pngpath = f"{path}/dim_{self.dim}/attribute_{self.attribute}/ctx_number_{self.context_number}/run_{nr+1}/fold_{i+1}"
                    filename = f'ppc_{alpha}_{beta[0]}_{beta[1]}_{beta[2]}_{beta[3]}_{beta[4]}'
                    plt.savefig(f"{pngpath}" + "/" + filename + ".png")
                plt.close()

                az.plot_trace(self.trace)
                if savepngs:
                    pngpath = f"{path}/dim_{self.dim}/attribute_{self.attribute}/ctx_number_{self.context_number}/run_{nr+1}/fold_{i+1}"
                    filename = f'trace_{alpha}_{beta[0]}_{beta[1]}_{beta[2]}_{beta[3]}_{beta[4]}'
                    plt.savefig(f"{pngpath}" + "/" + filename + ".png")
                plt.close()

        # Return the final predictions, and the RMSE calculations
        return running_R, results

    def rmse(self, test_data, predicted):
        """Calculate root mean squared error.
        Ignoring missing values in the test data.
        """
        I = ~np.isnan(test_data)  # indicator for missing values
        N = I.sum()  # number of non-missing values
        sqerror = abs(test_data - predicted) ** 2  # squared error array
        mse = sqerror[I].sum() / N  # mean squared error
        return np.sqrt(mse)  # RMSE

