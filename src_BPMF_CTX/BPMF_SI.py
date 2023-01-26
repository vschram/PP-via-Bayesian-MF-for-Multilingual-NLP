# Based on: https://docs.pymc.io/en/stable/pymc-examples/examples/case_studies/probabilistic_matrix_factorization.html
# Based on: https://gist.github.com/macks22/00a17b1d374dfc267a9a

import logging
import time
import numpy as np
import pymc3 as pm
import arviz as az
import theano
import pandas as pd
import matplotlib.pyplot as plt
import theano.tensor as t


# Enable on-the-fly graph computations, but ignore
# absence of intermediate test values.
theano.config.compute_test_value = "ignore"

class BPMF:
    """Probabilistic Matrix Factorization model using pymc3."""

    def __init__(self, mod, logger, other_params, data, src_si_len, tgt_si_len, lang_pair_si_len, dim=2, alpha=2, std=0.01, X=None, Y=None, Z_orig=None):

        """Build the Bayesian Matrix Factorization model using pymc3.

        :param np.ndarray data: The training data to use for learning the model.
        :param int dim: Dimensionality of the model; number of latent factors.
        :param int alpha: Fixed precision for the likelihood function.
        :param float std: Amount of noise to use for model initialization.
        :param (tuple of int) bounds: (lower, upper) bound of ratings.
            These bounds will simply be used to cap the estimates produced for R.

        """
        self.R = np.array(data)
        self.score_dict = {'BLEU': "WIKI-MT", "Muse": "BLI-Muse", "Vecmap": "BLI-Vecmap"}
        self.score = mod
        self.dim = dim
        self.attribute = other_params[5]
        self.context_number = other_params[6]
        self.alpha = alpha
        self.std = np.sqrt(1.0 / alpha)
        self.data = self.R.copy()
        self.logger = logger

        """Build the modified BPMF model using pymc3. The original model uses
        Wishart priors on the covariance matrices. Unfortunately, the Wishart
        distribution in pymc3 is currently not suitable for sampling. This
        version decomposes the covariance matrix into:
            diag(sigma) \dot corr_matrix \dot diag(std).
        We use uniform priors on the standard deviations (sigma) and LKJCorr
        priors on the correlation matrices (corr_matrix):
            sigma ~ Uniform
            corr_matrix ~ LKJCorr(n=1, p=dim)
        """
        n, m = self.data.shape
        beta_0 = 1  # scaling factor for lambdas; unclear on its use

        # Mean value imputation on training data.
        a = np.ma.masked_where(self.data == 0, self.data)
        # Perform mean value imputation
        nan_mask = np.isnan(self.data)
        self.data[nan_mask] = self.data[~nan_mask].mean()

        # We will use separate priors for sigma and correlation matrix.
        # In order to convert the upper triangular correlation values to a
        # complete correlation matrix, we need to construct an index matrix:
        # n_elem = np.int(np.ceil(dim * (dim - 1) / 2))
        # tri_index = np.zeros([dim, dim], dtype=int)
        # tri_index[np.triu_indices(dim, k=1)] = np.arange(n_elem)
        # tri_index[np.triu_indices(dim, k=1)[::-1]] = np.arange(n_elem)

        if other_params[0]:
            logger.info('*' * 20)
            logger.info('building the BPMF model')

        self.lang_pair_si_len = lang_pair_si_len
        self.Z = []
        for i in range(0, self.lang_pair_si_len, 1):
            Z = Z_orig[i].fillna(0)
            Z = Z.to_numpy()
            Z[nan_mask] = Z[~nan_mask].mean()
            self.Z.append(Z)

        # We will use separate priors for sigma and correlation matrix.
        # In order to convert the upper triangular correlation values to a
        # complete correlation matrix, we need to construct an index matrix:

        with pm.Model() as bpmf:
            # based on https://stats.stackexchange.com/questions/77038/covariance-matrix-proposal-distribution/101408#101408

            # ---------------------------------------------------------------------------
            # Prior and hyperpriors for W
            # ---------------------------------------------------------------------------
            # sigma_w = pm.Uniform('sigma_w', shape=dim)
            sigma_w = pm.HalfNormal.dist(sd=1)
            chol_w, corr, sigmas = pm.LKJCholeskyCov("chol_cov_matrix_w", n=dim, eta=1.0, sd_dist=sigma_w,
                                                     compute_corr=True, testval=np.random.randn(dim) * std)
            cov_matrix_w = pm.Deterministic('cov_matrix_w', t.dot(chol_w, chol_w.T))
            lambda_w = t.nlinalg.matrix_inverse(cov_matrix_w)

            # ----------------------------------------------------------------------------
            # Some alternatives, deprecated:
            # ---------------------------------------------------------------------------
            # Specify user feature matrix
            # sigma_w = pm.Uniform('sigma_w', shape=dim)
            # corr_triangle_w = pm.LKJCorr(
            #    'corr_u', n=1, p=dim,
            #    testval=np.random.randn(n_elem) * std)

            # corr_matrix_w = corr_triangle_w[tri_index]
            # corr_matrix_w = t.fill_diagonal(corr_matrix_w, 1)
            # cov_matrix_w = t.diag(sigma_w).dot(corr_matrix_w.dot(t.diag(sigma_w)))
            # lambda_w = t.nlinalg.matrix_inverse(cov_matrix_w)

            mu_w = pm.Normal(
                'mu_w', mu=0, tau=beta_0 * t.diag(lambda_w), shape=dim,
                testval=np.random.randn(dim) * std)
            W = pm.MvNormal(
                'W', mu=mu_w, tau=lambda_w * np.eye(dim), shape=(n, dim),
                testval=np.random.randn(n, dim) * std)

            # ---------------------------------------------------------------------------
            # Prior and hyperpriors for H
            # ---------------------------------------------------------------------------
            # sigma_h = pm.Uniform('sigma_w', shape=dim)
            sigma_h = pm.HalfNormal.dist(sd=1)
            chol_h, corr, sigmas = pm.LKJCholeskyCov("chol_matrix_h", n=dim, eta=1.0, sd_dist=sigma_h,
                                                     compute_corr=True, testval=np.random.randn(dim) * std)
            cov_matrix_h = pm.Deterministic('cov_matrix_h', t.dot(chol_h, chol_h.T))
            lambda_h = t.nlinalg.matrix_inverse(cov_matrix_h)

            # ----------------------------------------------------------------------------
            # Some alternatives, deprecated:
            # ---------------------------------------------------------------------------
            # sigma_h = pm.Uniform('sigma_h', shape=dim)
            # corr_triangle_h = pm.LKJCorr(
            #    'corr_h', n=1, p=dim,
            #    testval=np.random.randn(n_elem) * std)

            # corr_matrix_h = corr_triangle_h[tri_index]
            # corr_matrix_h = t.fill_diagonal(corr_matrix_h, 1)
            # cov_matrix_h = t.diag(sigma_h).dot(corr_matrix_h.dot(t.diag(sigma_h)))
            # lambda_h = t.nlinalg.matrix_inverse(cov_matrix_h)

            mu_v = pm.Normal(
                'mu_h', mu=0, tau=beta_0 * t.diag(lambda_h), shape=dim,
                testval=np.random.randn(dim) * std)
            H = pm.MvNormal(
                'H', mu=mu_v, tau=lambda_h * np.eye(dim), shape=(m, dim),
                testval=np.random.randn(m, dim) * std)

            a_var = 0.001
            if self.lang_pair_si_len == 15:
                a0 = pm.Normal('a0', mu=0, tau=1/a_var, testval=np.random.randn())
                a1 = pm.Normal('a1', mu=0, tau=1 / a_var, testval=np.random.randn())
                a2 = pm.Normal('a2', mu=0, tau=1 / a_var, testval=np.random.randn())
                a3 = pm.Normal('a3', mu=0, tau=1 / a_var, testval=np.random.randn())
                a4 = pm.Normal('a4', mu=0, tau=1 / a_var, testval=np.random.randn())
                a5 = pm.Normal('a5', mu=0, tau=1 / a_var, testval=np.random.randn())
                a6 = pm.Normal('a6', mu=0, tau=1 / a_var, testval=np.random.randn())
                a7 = pm.Normal('a7', mu=0, tau=1 / a_var, testval=np.random.randn())
                a8 = pm.Normal('a8', mu=0, tau=1 / a_var, testval=np.random.randn())
                a9 = pm.Normal('a9', mu=0, tau=1 / a_var, testval=np.random.randn())
                a10 = pm.Normal('a10', mu=0, tau=1 / a_var, testval=np.random.randn())
                a11 = pm.Normal('a11', mu=0, tau=1 / a_var, testval=np.random.randn())
                a12 = pm.Normal('a12', mu=0, tau=1 / a_var, testval=np.random.randn())
                a13 = pm.Normal('a13', mu=0, tau=1 / a_var, testval=np.random.randn())
                a14 = pm.Normal('a14', mu=0, tau=1 / a_var, testval=np.random.randn())

                mu_R = pm.Deterministic('mu_R', W @ H.T + a0 * self.Z[0] + a1 * self.Z[1] + a2 * self.Z[2]
                                        + a3 * self.Z[3] + a4 * self.Z[4] + a5 * self.Z[5] + a6 * self.Z[6]
                                        + a7 * self.Z[7] + a8 * self.Z[8] + a9 * self.Z[9] + a10 * self.Z[10]
                                        + a11 * self.Z[11] + a12 * self.Z[12] + a13 * self.Z[13] + a14 * self.Z[14]
                                        )

            elif self.lang_pair_si_len == 12:
                a0 = pm.Normal('a0', mu=0, tau=1/a_var, testval=np.random.randn())
                a1 = pm.Normal('a1', mu=0, tau=1 / a_var, testval=np.random.randn())
                a2 = pm.Normal('a2', mu=0, tau=1 / a_var, testval=np.random.randn())
                a3 = pm.Normal('a3', mu=0, tau=1 / a_var, testval=np.random.randn())
                a4 = pm.Normal('a4', mu=0, tau=1 / a_var, testval=np.random.randn())
                a5 = pm.Normal('a5', mu=0, tau=1 / a_var, testval=np.random.randn())
                a6 = pm.Normal('a6', mu=0, tau=1 / a_var, testval=np.random.randn())
                a7 = pm.Normal('a7', mu=0, tau=1 / a_var, testval=np.random.randn())
                a8 = pm.Normal('a8', mu=0, tau=1 / a_var, testval=np.random.randn())
                a9 = pm.Normal('a9', mu=0, tau=1 / a_var, testval=np.random.randn())
                a10 = pm.Normal('a10', mu=0, tau=1 / a_var, testval=np.random.randn())
                a11 = pm.Normal('a11', mu=0, tau=1 / a_var, testval=np.random.randn())


                mu_R = pm.Deterministic('mu_R', W @ H.T + a0 * self.Z[0] + a1 * self.Z[1] + a2 * self.Z[2]
                                        + a3 * self.Z[3] + a4 * self.Z[4] + a5 * self.Z[5] + a6 * self.Z[6]
                                        + a7 * self.Z[7] + a8 * self.Z[8] + a9 * self.Z[9] + a10 * self.Z[10]
                                        + a11 * self.Z[11]
                                        )

            elif self.lang_pair_si_len == 22:
                a0 = pm.Normal('a0', mu=0, tau=1 / a_var, testval=np.random.randn())
                a1 = pm.Normal('a1', mu=0, tau=1 / a_var, testval=np.random.randn())
                a2 = pm.Normal('a2', mu=0, tau=1 / a_var, testval=np.random.randn())
                a3 = pm.Normal('a3', mu=0, tau=1 / a_var, testval=np.random.randn())
                a4 = pm.Normal('a4', mu=0, tau=1 / a_var, testval=np.random.randn())
                a5 = pm.Normal('a5', mu=0, tau=1 / a_var, testval=np.random.randn())
                a6 = pm.Normal('a6', mu=0, tau=1 / a_var, testval=np.random.randn())
                a7 = pm.Normal('a7', mu=0, tau=1 / a_var, testval=np.random.randn())
                a8 = pm.Normal('a8', mu=0, tau=1 / a_var, testval=np.random.randn())
                a9 = pm.Normal('a9', mu=0, tau=1 / a_var, testval=np.random.randn())
                a10 = pm.Normal('a10', mu=0, tau=1 / a_var, testval=np.random.randn())
                a11 = pm.Normal('a11', mu=0, tau=1 / a_var, testval=np.random.randn())
                a12 = pm.Normal('a12', mu=0, tau=1 / a_var, testval=np.random.randn())
                a13 = pm.Normal('a13', mu=0, tau=1 / a_var, testval=np.random.randn())
                a14 = pm.Normal('a14', mu=0, tau=1 / a_var, testval=np.random.randn())
                a15 = pm.Normal('a15', mu=0, tau=1 / a_var, testval=np.random.randn())
                a16 = pm.Normal('a16', mu=0, tau=1 / a_var, testval=np.random.randn())
                a17 = pm.Normal('a17', mu=0, tau=1 / a_var, testval=np.random.randn())
                a18 = pm.Normal('a18', mu=0, tau=1 / a_var, testval=np.random.randn())
                a19 = pm.Normal('a19', mu=0, tau=1 / a_var, testval=np.random.randn())
                a20 = pm.Normal('a20', mu=0, tau=1 / a_var, testval=np.random.randn())
                a21 = pm.Normal('a21', mu=0, tau=1 / a_var, testval=np.random.randn())

                mu_R = pm.Deterministic('mu_R', W @ H.T + a0 * self.Z[0] + a1 * self.Z[1] + a2 * self.Z[2]
                                        + a3 * self.Z[3] + a4 * self.Z[4] + a5 * self.Z[5] + a6 * self.Z[6]
                                        + a7 * self.Z[7] + a8 * self.Z[8] + a9 * self.Z[9] + a10 * self.Z[10]
                                        + a11 * self.Z[11] + a12 * self.Z[12] + a13 * self.Z[13] + a14 * self.Z[14]
                                        + a15 * self.Z[15] + a16 * self.Z[16] + a17 * self.Z[17]
                                        + a18 * self.Z[18] + a19 * self.Z[19] + a20 * self.Z[20] + a21 * self.Z[21]
                                        )

            elif self.lang_pair_si_len == 21:
                a0 = pm.Normal('a0', mu=0, tau=1 / a_var, testval=np.random.randn())
                a1 = pm.Normal('a1', mu=0, tau=1 / a_var, testval=np.random.randn())
                a2 = pm.Normal('a2', mu=0, tau=1 / a_var, testval=np.random.randn())
                a3 = pm.Normal('a3', mu=0, tau=1 / a_var, testval=np.random.randn())
                a4 = pm.Normal('a4', mu=0, tau=1 / a_var, testval=np.random.randn())
                a5 = pm.Normal('a5', mu=0, tau=1 / a_var, testval=np.random.randn())
                a6 = pm.Normal('a6', mu=0, tau=1 / a_var, testval=np.random.randn())
                a7 = pm.Normal('a7', mu=0, tau=1 / a_var, testval=np.random.randn())
                a8 = pm.Normal('a8', mu=0, tau=1 / a_var, testval=np.random.randn())
                a9 = pm.Normal('a9', mu=0, tau=1 / a_var, testval=np.random.randn())
                a10 = pm.Normal('a10', mu=0, tau=1 / a_var, testval=np.random.randn())
                a11 = pm.Normal('a11', mu=0, tau=1 / a_var, testval=np.random.randn())
                a12 = pm.Normal('a12', mu=0, tau=1 / a_var, testval=np.random.randn())
                a13 = pm.Normal('a13', mu=0, tau=1 / a_var, testval=np.random.randn())
                a14 = pm.Normal('a14', mu=0, tau=1 / a_var, testval=np.random.randn())
                a15 = pm.Normal('a15', mu=0, tau=1 / a_var, testval=np.random.randn())
                a16 = pm.Normal('a16', mu=0, tau=1 / a_var, testval=np.random.randn())
                a17 = pm.Normal('a17', mu=0, tau=1 / a_var, testval=np.random.randn())
                a18 = pm.Normal('a18', mu=0, tau=1 / a_var, testval=np.random.randn())
                a19 = pm.Normal('a19', mu=0, tau=1 / a_var, testval=np.random.randn())
                a20 = pm.Normal('a20', mu=0, tau=1 / a_var, testval=np.random.randn())

                mu_R = pm.Deterministic('mu_R', W @ H.T + a0 * self.Z[0] + a1 * self.Z[1] + a2 * self.Z[2]
                                        + a3 * self.Z[3] + a4 * self.Z[4] + a5 * self.Z[5] + a6 * self.Z[6]
                                        + a7 * self.Z[7] + a8 * self.Z[8] + a9 * self.Z[9] + a10 * self.Z[10]
                                        + a11 * self.Z[11] + a12 * self.Z[12] + a13 * self.Z[13] + a14 * self.Z[14]
                                        + a15 * self.Z[15] + a16 * self.Z[16] + a17 * self.Z[17]
                                        + a18 * self.Z[18] + a19 * self.Z[19] + a20 * self.Z[20])

            elif self.lang_pair_si_len == 6:
                a0 = pm.Normal('a0', mu=0, tau=1 / a_var, testval=np.random.randn())
                a1 = pm.Normal('a1', mu=0, tau=1 / a_var, testval=np.random.randn())
                a2 = pm.Normal('a2', mu=0, tau=1 / a_var, testval=np.random.randn())
                a3 = pm.Normal('a3', mu=0, tau=1 / a_var, testval=np.random.randn())
                a4 = pm.Normal('a4', mu=0, tau=1 / a_var, testval=np.random.randn())
                a5 = pm.Normal('a5', mu=0, tau=1 / a_var, testval=np.random.randn())


                mu_R = pm.Deterministic('mu_R', W @ H.T + a0 * self.Z[0] + a1 * self.Z[1] + a2 * self.Z[2]
                                        + a3 * self.Z[3] + a4 * self.Z[4] + a5 * self.Z[5])


            # ---------------------------------------------------------------------------
            # Specify rating likelihood function
            # ---------------------------------------------------------------------------
            R = pm.Normal("R", mu=(mu_R)[~nan_mask], tau=self.alpha, observed=self.data[~nan_mask])

        #gv = pm.model_to_graphviz(bpmf)
        #gv.render(filename='bpmf_model', format='pdf')

        if other_params[0]:
            logger.info('*' * 20)
            logger.info('done building the BPMF model')
        self.model = bpmf

    def __str__(self):
        return self.name

    def find_map(self, method):
        """Find mode of posterior using L-BFGS-B optimization."""
        tstart = time.time()
        with self.model:
            self.logger.info("finding PMF MAP ")  # using L-BFGS-B optimization...
            self.map = pm.find_MAP(method=method)  # method="L-BFGS-B", "Powell"

        elapsed = int(time.time() - tstart)
        self.logger.info("found PMF MAP in %d seconds" % elapsed)
        return self.map

    def map(self):
        try:
            return self.map
        except:
            return self.find_map()

    # Draw MCMC samples.
    def draw_samples(self, **kwargs):
        # kwargs.setdefault("chains", 1)  # trace = pm.sample(1000, tune=1000)
        with self.model:
            self.trace = pm.sample(**kwargs)  # **kwargs,
        return self.trace

    def predict(self, W, H, a, fmtest, fmtrain):
        """Estimate R from the given values of U and V."""
        fm = []

        for i in range(0, self.lang_pair_si_len, 1):
            feats = fmtest[i].fillna(0) + fmtrain[i].fillna(0)
            fm.append(feats.to_numpy())

        if self.lang_pair_si_len == 15:
            features = a[0]*fm[0] + a[1]*fm[1] + a[2]*fm[2] + a[3]*fm[3] + a[4]*fm[4] + a[5]*fm[5] + a[6]*fm[6] \
                       + a[7] * fm[7] + a[8]*fm[8] + a[9]*fm[9] + a[10]*fm[10] + a[11]*fm[11] + a[12]*fm[12] + a[13]*fm[13] + a[14]*fm[14]

        elif self.lang_pair_si_len == 12:
            features = a[0]*fm[0] + a[1]*fm[1] + a[2]*fm[2] + a[3]*fm[3] + a[4]*fm[4] + a[5]*fm[5] + a[6]*fm[6] \
                       + a[7] * fm[7] + a[8]*fm[8] + a[9]*fm[9] + a[10]*fm[10] + a[11]*fm[11]

        elif self.lang_pair_si_len == 6:
            features = a[0]*fm[0] + a[1]*fm[1] + a[2]*fm[2] + a[3]*fm[3] + a[4]*fm[4] + a[5]*fm[5]


        elif self.lang_pair_si_len == 22:
            features = a[0]*fm[0] + a[1]*fm[1] + a[2]*fm[2] + a[3]*fm[3] + a[4]*fm[4] + a[5]*fm[5] + a[6]*fm[6] \
                       + a[7] * fm[7] + a[8]*fm[8] + a[9]*fm[9] + a[10]*fm[10] + a[11]*fm[11] + a[12]*fm[12] + a[13]*fm[13] + a[14]*fm[14] \
                       + a[15]*fm[15] + a[16]*fm[16] + a[17]*fm[17] + a[18]*fm[18] + a[19]*fm[19] + a[20]*fm[20] + a[21]*fm[21]

        elif self.lang_pair_si_len == 21:
            features = a[0]*fm[0] + a[1]*fm[1] + a[2]*fm[2] + a[3]*fm[3] + a[4]*fm[4] + a[5]*fm[5] + a[6]*fm[6] \
                       + a[7] * fm[7] + a[8]*fm[8] + a[9]*fm[9] + a[10]*fm[10] + a[11]*fm[11] + a[12]*fm[12] + a[13]*fm[13] + a[14]*fm[14] \
                       + a[15]*fm[15] + a[16]*fm[16] + a[17]*fm[17] + a[18]*fm[18] + a[19]*fm[19] + a[20]*fm[20]

        R = np.dot(W, H.T) + features
        n, m = R.shape
        sample_R = np.random.normal(R, self.std)
        return sample_R

    def eval_map(self, train, test):
        U = self.map["U"]
        V = self.map["V"]
        a = self.map["a"]

        # Make predictions and calculate RMSE on train & test sets.
        predictions = self.predict(U, V, a)
        train_rmse = rmse(train, predictions)
        test_rmse = rmse(test, predictions)
        overfit = test_rmse - train_rmse

        # Print report.
        print("PMF MAP training RMSE: %.5f" % train_rmse)
        print("PMF MAP testing RMSE:  %.5f" % test_rmse)
        print("Train/test difference: %.5f" % overfit)

        return test_rmse, train_rmse

    def norms(self, monitor=("U", "V"), ord="fro"):
        """Return norms of latent variables at each step in the
        sample trace. These can be used to monitor convergence
        of the sampler.
        """
        monitor = ("U", "V")
        norms = {var: [] for var in monitor}
        for sample in self.trace:
            for var in monitor:
                norms[var].append(np.linalg.norm(sample[var], ord))
        return norms

    def traceplot(self, draws, tune, chains, DIM, ALPHA, fold, run, model):
        """Plot Frobenius norms of U and V as a function of sample #."""
        trace_norms = self.norms()
        u_series = pd.Series(trace_norms["U"])
        v_series = pd.Series(trace_norms["V"])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        u_series.plot(kind="line", ax=ax1, grid=False, title=r"$\|U\|_{Fro}^2$ at Each Sample")
        v_series.plot(kind="line", ax=ax2, grid=False, title=r"$\|V\|_{Fro}^2$ at Each Sample")
        ax1.set_xlabel("Sample Number")
        ax2.set_xlabel("Sample Number")
        plt.suptitle(f'Run: {run}, Fold: {fold}, Model: {model}, Dim: {DIM}, Alpha: {ALPHA}, Chains: {chains}, Draws: {draws}, Tune: {tune}')
        return fig

    def running_rmse(self, fmtest, fmtrain, test_data, train_data,  nr, i, ii, path, savepngs, doplots, alpha, flag_test=0, burn_in=0):
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
            if self.lang_pair_si_len == 15:
                a = [sample["a0"], sample["a1"], sample["a2"], sample["a3"], sample["a4"], sample["a5"], sample["a6"],
                     sample["a7"], sample["a8"], sample["a9"], sample["a10"], sample["a11"], sample["a12"], sample["a13"],
                     sample["a14"]]

            elif self.lang_pair_si_len == 6:
                a = [sample["a0"], sample["a1"], sample["a2"], sample["a3"], sample["a4"], sample["a5"]]

            elif self.lang_pair_si_len == 12:
                a = [sample["a0"], sample["a1"], sample["a2"], sample["a3"], sample["a4"], sample["a5"], sample["a6"],
                     sample["a7"], sample["a8"], sample["a9"], sample["a10"], sample["a11"]]

            elif self.lang_pair_si_len == 22:
                a = [sample["a0"], sample["a1"], sample["a2"], sample["a3"], sample["a4"], sample["a5"], sample["a6"],
                     sample["a7"], sample["a8"], sample["a9"], sample["a10"], sample["a11"], sample["a12"], sample["a13"],
                     sample["a14"], sample["a15"], sample["a16"], sample["a17"], sample["a18"], sample["a19"], sample["a20"], sample["a21"]]

            elif self.lang_pair_si_len == 21:
                a = [sample["a0"], sample["a1"], sample["a2"], sample["a3"], sample["a4"], sample["a5"], sample["a6"],
                     sample["a7"], sample["a8"], sample["a9"], sample["a10"], sample["a11"], sample["a12"], sample["a13"],
                     sample["a14"], sample["a15"], sample["a16"], sample["a17"], sample["a18"], sample["a19"], sample["a20"]]

            sample_R = self.predict(sample["W"], sample["H"], a, fmtest, fmtrain)
            # deal with test data
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
                title=f'| Posterior Predictive | Run: {nr} | Fold: {i} | CV fold: {ii}|'
            )

            if savepngs:
                pngpath = f"{path}/dim_{self.dim}/attribute_{self.attribute}/ctx_number_{self.context_number}/run_{nr + 1}/fold_{i+1}"
                filename = f'{alpha}'
                plt.savefig(f"{pngpath}" + "/" + filename + ".png")
            # plt.show()
            plt.close()

            with self.model:
                ppc = pm.sample_posterior_predictive(self.trace, var_names=["W", "H", "R"], random_seed=2706)
                az.plot_ppc(az.from_pymc3(posterior_predictive=ppc, model=self.model))
                if savepngs:
                    pngpath = f"{path}/dim_{self.dim}/attribute_{self.attribute}/ctx_number_{self.context_number}/run_{nr + 1}/fold_{i + 1}"
                    filename = f'ppc_{alpha}'
                    plt.savefig(f"{pngpath}" + "/" + filename + ".png")
                plt.close()

                az.plot_trace(self.trace)
                if savepngs:
                    pngpath = f"{path}/dim_{self.dim}/attribute_{self.attribute}/ctx_number_{self.context_number}/run_{nr + 1}/fold_{i + 1}"
                    filename = f'trace_{alpha}'
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

