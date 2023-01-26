import numpy as np
np.random.seed(2021)
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt


class MF():

    def __init__(self, logger, R, other_params, K, alpha, beta, iterations, dev_samples, X=None, Y=None, Z=None, src_si_len=0, \
                 tgt_si_len=0, lang_pair_si_len=0, src_index=None, tgt_index=None, model=None, num_running=0):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - R (ndarray)                  : src-tgt language rating matrix
        - K (int)                      : number of latent dimensions
        - alpha (float)                : learning rate
        - beta (float)                 : regularization parameter
        - X (dict)                     : source language side information
        - Y (dict)                     : target language side information
        - Z (dict)                     : language pair side information
        - src_si_len(int)              : source language side information length
        - tgt_si_len(int)              : target language side information length
        - lang_pair_si_len(int)        : language pair side information length
        """

        self.R = np.array(R)
        self.Prediction = deepcopy(self.R)
        self.src_langs = R.index.tolist()
        self.tgt_langs = R.columns.tolist()
        self.num_src, self.num_tgt = R.shape
        self.K = K
        self.alpha = alpha
        # BETA = [reg_w, reg_h, reg_x, reg_y, reg_z, reg_bias_s, reg_bias_l]
        self.beta_h = beta[1]
        self.beta_w = beta[0]
        self.beta_x = beta[2]
        self.beta_y = beta[3]
        self.beta_z = beta[4]
        self.beta_s = beta[5]
        self.beta_t = beta[6]
        self.iterations = iterations
        self.X = X
        self.Y = Y
        self.Z = Z
        self.src_si_len = src_si_len
        self.tgt_si_len = tgt_si_len
        self.lang_pair_si_len = lang_pair_si_len
        self.dev_samples = dev_samples
        self.src_index = src_index
        self.tgt_index = tgt_index
        self.model = model
        self.score_dict = {'BLEU': "WIKI-MT", "Muse": "BLI-Muse", "Vecmap": "BLI-Vecmap"}
        self.traing_error_log = []
        self.dev_error_log = []
        self.enable_print = other_params[0]
        self.doplots = other_params[1]
        self.debug_reg_z = other_params[7]
        self.attribute = other_params[5]
        self.context_number = other_params[6]
        self.logger = logger
        self.eval_on_dev = other_params[8]
        if other_params[9]:
            c = np.exp(1)
            # linear method
            self.num_scores_source = np.count_nonzero(self.R, axis=1) + c
            self.num_scores_target = np.count_nonzero(self.R, axis=0) + c
        else:
            self.num_scores_source = np.ones(self.num_src)
            self.num_scores_target = np.ones(self.num_tgt)

        self.lr_decay = other_params[10]




    def train(self):
        # Initialize user and item latent feature matrice
        self.W = np.random.normal(scale=1. / self.K, size=(self.num_src, self.K))
        self.H = np.random.normal(scale=1. / self.K, size=(self.num_tgt, self.K))

        # Initialize side information's parameter if necesary
        if self.X and self.src_si_len:
            self.A = np.random.normal(scale=1. / self.src_si_len, size=self.src_si_len)
        if self.Y and self.tgt_si_len:
            self.B = np.random.normal(scale=1. / self.tgt_si_len, size=self.tgt_si_len)
        if self.Z and self.lang_pair_si_len:
            if self.debug_reg_z:
                self.C = np.ones(self.lang_pair_si_len)
            else:
                self.C = np.random.normal(scale=1. / self.lang_pair_si_len, size=self.lang_pair_si_len)

        # Initialize the biases
        # the biases of users and items are initilized as 0
        # the bias of rating is initilized as mean value
        self.b_s = np.zeros(self.num_src)
        self.b_t = np.zeros(self.num_tgt)
        if self.debug_reg_z:
            self.b = np.array([0])
        else:
            self.b = np.mean(self.R[np.where(self.R != 0)])

        # Create a list of training samples (where rating > 0)
        self.samples = []
        for i in range(self.num_src):
            for j in range(self.num_tgt):
                if self.R[i, j] > 0:
                    cur_tuple = [i, j, self.R[i, j]]
                    src_lang = self.src_langs[i]
                    tgt_lang = self.tgt_langs[j]
                    if self.X:
                        if src_lang in self.X.keys():
                            cur_tuple.append(self.X[src_lang])
                        else:
                            raise KeyError
                    if self.Y:
                        if tgt_lang in self.Y.keys():
                            cur_tuple.append(self.Y[src_lang])
                        else:
                            raise KeyError
                    if self.Z:
                        if src_lang + "_" + tgt_lang in self.Z.keys():
                            cur_tuple.append(self.Z[src_lang + "_" + tgt_lang])
                        else:
                            raise KeyError
                    self.samples.append(tuple(cur_tuple))

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        development_process = []
        if self.eval_on_dev:
            self.logger.info("----"*10)
            self.logger.info("Hyperparam optimization")
            self.logger.info("----" * 10)
        else:
            self.logger.info("****"*10)
            self.logger.info("Training on full data set (train+dev)")
            self.logger.info("****" * 10)

        for i in range(self.iterations):
            # shuffle training samples
            if self.lr_decay:
                if (i + 1) % 100 == 0:
                    decay = 0.001
                    self.alpha *= (1. / (1. + decay * (i+1)))
                    print(self.alpha)
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            if self.eval_on_dev:
                dev_mse = self.evaluate_testing(self.dev_samples, self.src_index, self.tgt_index, self.model)
                development_process.append((i, dev_mse))
            else:
                development_process = 'Currently in test mode'
                dev_mse = 'Currently in test mode'

            if self.enable_print == 1:
                if (i + 1) % 100 == 0:
                    if self.eval_on_dev:
                        self.traing_error_log.append((i, mse))
                        self.logger.info("Iteration: %d ; rmse train error = %.4f" % (i + 1, mse))
                        self.dev_error_log.append((i, dev_mse))
                        self.logger.info("Iteration: %d ; rmse eval error = %.4f" % (i + 1, dev_mse))
                    else:
                        self.traing_error_log.append((i, mse))
                        self.logger.info("Iteration: %d ; rmse train error = %.4f" % (i + 1, mse))
                        self.dev_error_log = 'Currently in test mode'

            if np.isnan(mse) or np.isinf(mse):
                self.logger.info('+-+-'*20)
                self.logger.info('NAN detected. Exploding or vanishing gradient. Terminate training iterations.')
                self.logger.info('+-+-' * 20)
                break

        training_process = [training_process, development_process]

        return training_process

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        #         predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x][y] - self.Prediction[x][y], 2)
        return np.sqrt(error / len(xs))

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for sample in self.samples:
            i, j, r = sample[0], sample[1], sample[2]
            # Computer prediction and error
            prediction = self.get_rating(sample)
            self.Prediction[i][j] = prediction
            e = (r - prediction)

            #BETA = [reg_w, reg_h, reg_x, reg_y, reg_z, reg_bias_s, reg_bias_l]

            # Update biases
            self.b_s[i] += self.alpha * (e - self.beta_s/self.num_scores_source[i] * self.b_s[i])
            self.b_t[j] += self.alpha * (e - self.beta_t/self.num_scores_target[j] * self.b_t[j])

            # Update user and item latent feature matrices
            self.W[i, :] += self.alpha * (e * self.H[j, :] - self.beta_w/self.num_scores_source[i] * self.W[i, :])
            self.H[j, :] += self.alpha * (e * self.W[i, :] - self.beta_h/self.num_scores_target[j] * self.H[j, :])

            # Update side information parameter if necessary
            cur_index = 3
            if self.X:
                x = np.array(sample[cur_index], dtype=np.float64)
                cur_index += 1
                self.A += self.alpha * (e * x - self.beta_x/self.num_scores_source[i] * self.A)
            if self.Y:
                y = np.array(sample[cur_index], dtype=np.float64)
                cur_index += 1
                self.B += self.alpha * (e * y - self.beta_y/self.num_scores_target[j] * self.B)
            if self.Z:
                z = np.array(sample[cur_index], dtype=np.float64)
                if self.debug_reg_z:
                    self.C = np.ones(self.lang_pair_si_len)
                else:
                    self.C += self.alpha * (e * z - self.beta_z * self.C)


    def get_rating(self, sample):
        """
        Get the predicted rating of sample
        """
        i, j, r = sample[0], sample[1], sample[2]
        prediction = self.b + self.b_s[i] + self.b_t[j] + self.W[i, :].dot(self.H[j, :].T)
        cur_index = 3
        if self.X:
            x = sample[cur_index]
            cur_index += 1
            prediction += self.A.dot(x.T)
        if self.Y:
            y = sample[cur_index]
            cur_index += 1
            prediction += self.B.dot(y.T)
        if self.Z:
            z = sample[cur_index]
            prediction += self.C.dot(z.T)

        return prediction

    def evaluate_testing(self, test_data, src_index_name, tgt_index_name, score_index_name):
        """
        Predict the score for testing data
        """
        rmse = 0.0
        for record in test_data.iterrows():
            record = record[1]
            src_lang = record[src_index_name]
            tgt_lang = record[tgt_index_name]
            src_lang_index = self.src_langs.index(src_lang)
            tgt_lang_index = self.tgt_langs.index(tgt_lang)
            score = record[score_index_name]
            cur_tuple = [src_lang_index, tgt_lang_index, score]
            if self.X:
                if src_lang in self.X.keys():
                    cur_tuple.append(self.X[src_lang])
                else:
                    raise KeyError
            if self.Y:
                if tgt_lang in self.Y.keys():
                    cur_tuple.append(self.Y[src_lang])
                else:
                    raise KeyError
            if self.Z:
                if src_lang + "_" + tgt_lang in self.Z.keys():
                    cur_tuple.append(self.Z[src_lang + "_" + tgt_lang])
                else:
                    raise KeyError
            prediction = self.get_rating(tuple(cur_tuple))
            rmse += (prediction - score) * (prediction - score)

        self.test_rmse = np.sqrt(rmse / len(test_data))
        return self.test_rmse

    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P, Q, A, B, and C
        """
        res = deepcopy(self.R)
        for i in range(self.num_src):
            for j in range(self.num_tgt):
                src_lang = self.src_langs[i]
                tgt_lang = self.tgt_langs[j]
                res[i][j] = self.b + self.b_s[i] + self.b_t[j] + self.W[i, :].dot(self.H[j, :].T)
                if self.X and src_lang in self.X.keys():
                    x = self.X[src_lang]
                    res[i][j] += self.A.dot(x.T)
                if self.Y and tgt_lang in self.Y.keys():
                    y = self.X[tgt_lang]
                    res[i][j] += self.A.dot(x.T)
                if self.Z and src_lang + "_" + tgt_lang in self.Z.keys():
                    z = self.X[src_lang + "_" + tgt_lang]
                    res[i][j] += self.A.dot(x.T)
        return self.b + self.b_s[:, np.newaxis] + self.b_t[np.newaxis:, ] + self.W.dot(self.H.T)

    def draw_error_curve(self, nr, i, ii, path, savepngs, doplots, alpha, beta):

        iters = []
        train_loss = []
        dev_loss = []

        for item in self.traing_error_log:
            iters.append(item[0])
            train_loss.append(item[1])

        for item in self.dev_error_log:
            dev_loss.append(item[1])

        if doplots:
            plt.plot(iters, train_loss, 'b', label='train RMSE')
            plt.plot(iters, dev_loss, 'r', label='dev RMSE')

            plt.legend()
            plt.xlabel('Iteration')
            plt.grid(True)
            plt.ylabel('RMSE')

            if self.debug_reg_z:
                title = f'| {self.score_dict[self.model]} | Run: {nr} | Fold: {i} | CV fold: {ii} | Test RMSE: {np.round(self.test_rmse[0], 2)} |'
            else:
                title = f'| {self.score_dict[self.model]} | Run: {nr} | Fold: {i} | CV fold: {ii} | Test RMSE: {np.round(self.test_rmse, 2)} |'

            plt.title(title)

            if savepngs:
                pngpath = f"{path}/dim_{self.K}/attribute_{self.attribute}/ctx_number_{self.context_number}/run_{nr+1}/fold_{i+1}/cv_fold_{ii+1}"
                filename = f'{alpha}_{beta[0]}_{beta[1]}_{beta[2]}_{beta[3]}_{beta[4]}_{beta[5]}_{beta[6]}_{self.iterations}'
                plt.savefig(f"{pngpath}" + "/" + filename + ".png")
            #plt.show()
            plt.close()