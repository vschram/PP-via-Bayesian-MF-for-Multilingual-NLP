''# main.py
import argparse
import os
import sys


def parse_args(debug=0):
    # Create the parser
    my_parser = argparse.ArgumentParser(description='Bilingual Performance Prediction - Matrix Factorization')

    if debug:

        # -----------------------------
        # Sort of fixed
        # -----------------------------
        my_parser.add_argument('--runs', default=2, type=int, help="Number of epochs, i.e. runs through the datas set")
        my_parser.add_argument('--folds', default=5, type=float, help="folds in CV, or outer folds in NCV")
        my_parser.add_argument('--split', default='NCV_lolo_source', type=str, help="|CV|: Cross Validation, w/o hyperparameter optimization; "
                                                                        "|NCV|: Nested CV, used for hyperparameter optimization; "
                                                                        "|NCV_NLPerf|: Dataset as in NLPerf."
                                                                        "|NCV_lolo_source|: Leave one language out."
                                                                        "|NCV_lolo_target|: Leave one language out.")
        my_parser.add_argument('--std', default=0.1, type=float, help="Amount of noise to use for model initialization")

        # -----------------------------
        # Different cases to analyse
        # -----------------------------
        my_parser.add_argument('--task', default="tsfparsing_MF", type=str, help="Tasks: wiki_MF, bli_MF, tsfpo, tsfel, tsfmt, tsfparsing")
        my_parser.add_argument('--model', default='BPMF', type=str, help="BPMF")
        my_parser.add_argument('--dim', default=1, type=int, help="Number of latent dimensions")
        my_parser.add_argument('--attribute', default="csi", type=str, help="Attributes to use: choose si, ctx, csi")
        my_parser.add_argument('--context', default=21, type=int, help="Available context for wiki MT:"
                                                                       "|1-15|: each feature individually"
                                                                       "|16|: Uriel lang. features,"
                                                                       "|17|: BLEU scores,"
                                                                       "|18|: 6 most correlated features,"
                                                                       "|19|: 11 most correlated features,"
                                                                       "|20|: 4 features, uncorrelated with BLEU,"
                                                                       "|21|: All avaiable features (w/o BLEU)"
                                                                       "BLI: No need to choose, just language distance "
                                                                       "features available.")

        # -----------------------------
        # hyperparam optimization for model selection:
        # -----------------------------
        my_parser.add_argument('--alpha', default=[2], type=int, help="Fixed precision for the likelihood function")

        # -----------------------------
        # MCMC params:
        # -----------------------------
        my_parser.add_argument('--chains', default=1, type=int, help="Markov Chains")
        my_parser.add_argument('--tune', default=10, type=int, help="Tune params of MC")
        my_parser.add_argument('--draws', default=10, type=int, help="Draws out of the markov chain")
        my_parser.add_argument('--burn', default=0, type=int, help="Burn in")

        # -----------------------------
        # Some other things:
        # -----------------------------
        my_parser.add_argument('--savepngs', default=False, type=bool, help="Save plots as pngs")
        my_parser.add_argument('--doplots', default=False, type=bool, help="Do plots")
        my_parser.add_argument('--enable_log', default=1, type=int, help="Print debugging messages")
        my_parser.add_argument('--diff_reg', default=0, type=int, help="Normalize regularization based on items given, smaller for more information. Higher for less.")
        my_parser.add_argument('--lr_decay', default=0, type=int, help="Print debugging messages")
        my_parser.add_argument('--uncertainty_analysis', default=1, type=int, help="Analyse uncertainty in R values")




    else:


        # -----------------------------
        # Sort of fixed
        # -----------------------------
        my_parser.add_argument('--runs', default=2, type=int, help="Number of epochs, i.e. runs through the datas set")
        my_parser.add_argument('--folds', default=5, type=float, help="folds in CV, or outer folds in NCV")
        my_parser.add_argument('--split', default='NCV_NLPerf', type=str, help="|CV|: Cross Validation, w/o hyperparameter optimization; "
                                                                        "|NCV|: Nested CV, used for hyperparameter optimization; "
                                                                        "|NCV_NLPerf|: Dataset as in NLPerf."
                                                                        "|NCV_lolo_source|: Leave one language out."
                                                                        "|NCV_lolo_target|: Leave one language out.")
        my_parser.add_argument('--std', default=0.1, type=float, help="Amount of noise to use for model initialization")

        # -----------------------------
        # Different cases to analyse
        # -----------------------------
        my_parser.add_argument('--task', default="tsfparsing_MF", type=str, help="Tasks: wiki_MF, bli_MF, tsfpos, tsfel, tsfmt, tsfparsing")
        my_parser.add_argument('--model', default='BPMF', type=str, help="BPMF")
        my_parser.add_argument('--dim', default=30, type=int, help="Number of latent dimensions")
        my_parser.add_argument('--attribute', default="ctx", type=str, help="Attributes to use: choose si, ctx, csi")
        my_parser.add_argument('--context', default=21, type=int, help="Available context for wiki MT:"
                                                                       "|1-15|: each feature individually"
                                                                       "|16|: Uriel lang. features,"
                                                                       "|17|: BLEU scores,"
                                                                       "|18|: 6 most correlated features,"
                                                                       "|19|: 11 most correlated features,"
                                                                       "|20|: 4 features, uncorrelated with BLEU,"
                                                                       "|21|: All avaiable features (w/o BLEU)"
                                                                       "BLI: No need to choose, just language distance "
                                                                       "features available.")

        # -----------------------------
        # hyperparam optimization for model selection:
        # -----------------------------
        my_parser.add_argument('--alpha', default=[2], type=int, help="Fixed precision for the likelihood function")

        # -----------------------------
        # MCMC params:
        # -----------------------------
        my_parser.add_argument('--chains', default=2, type=int, help="Markov Chains")
        my_parser.add_argument('--tune', default=1000, type=int, help="Tune params of MC")
        my_parser.add_argument('--draws', default=2000, type=int, help="Draws out of the markov chain")
        my_parser.add_argument('--burn', default=1000, type=int, help="Burn in")

        # -----------------------------
        # Some other things:
        # -----------------------------
        my_parser.add_argument('--savepngs', default=False, type=bool, help="Save plots as pngs")
        my_parser.add_argument('--doplots', default=False, type=bool, help="Do plots")
        my_parser.add_argument('--enable_log', default=1, type=int, help="Print debugging messages")
        my_parser.add_argument('--diff_reg', default=0, type=int, help="Normalize regularization based on items given, smaller for more information. Higher for less.")
        my_parser.add_argument('--lr_decay', default=0, type=int, help="Print debugging messages")
        my_parser.add_argument('--uncertainty_analysis', default=1, type=int, help="Analyse uncertainty in R values")



    args = my_parser.parse_args()

    return args