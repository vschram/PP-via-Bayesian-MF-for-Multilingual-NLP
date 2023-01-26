# main.py
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
        my_parser.add_argument('--iterations', default=20, type=int, help="Number of iterations of SGD")
        my_parser.add_argument('--folds', default=5, type=float, help="folds in CV, or outer folds in NCV")
        my_parser.add_argument('--split', default='NCV_lolo_source', type=str, help="|CV|: Cross Validation, w/o hyperparameter optimization; "
                                                                        "|NCV|: Nested CV, used for hyperparameter optimization; "
                                                                        "|NCV_NLPerf|: Dataset as in NLPerf."
                                                                        "|NCV_lolo_source|: Leave one language out."
                                                                        "|NCV_lolo_target|: Leave one language out.")


        # -----------------------------
        # Different cases to analyse
        # -----------------------------
        my_parser.add_argument('--task', default="wiki_MF", type=str, help="If only MF wanted, set: task_MF"
                                                                           "If MF with context wanted, set: task_aaMF (based on attribute aware MF)"
                                                                           "Available tasks:"
                                                                           "wiki, bli, wiki, bli, tsfpos, tsfel, tsfmt, tsfparsing")
        my_parser.add_argument('--dim', default=2, type=int, help="Number of latent dimensions")
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
        my_parser.add_argument('--lr', default=[0.01], type=float, help="Learning rate alpha")
        my_parser.add_argument('--reg_h', default=[0.001], type=float, help="Regularization param target languages")
        my_parser.add_argument('--reg_w', default=[0.001], type=float, help="Regularization param source languages")
        my_parser.add_argument('--reg_x', default=[0.00001], type=float, help="Regularization param attribute_source")
        my_parser.add_argument('--reg_y', default=[0.00001], type=float, help="Regularization param attribute_language")
        my_parser.add_argument('--reg_z', default=[1], type=float, help="Regularization param context matrix")  #0.00001
        my_parser.add_argument('--reg_bias_s', default=[0.01], type=float, help="Regularization param bias source language")
        my_parser.add_argument('--reg_bias_t', default=[0.01], type=float, help="Regularization param bias target language")

        # -----------------------------
        # Some other things:
        # -----------------------------
        my_parser.add_argument('--savepngs', default=False, type=bool, help="Save plots as pngs")
        my_parser.add_argument('--doplots', default=False, type=bool, help="Do plots")
        my_parser.add_argument('--enable_log', default=1, type=int, help="Print debugging messages")
        my_parser.add_argument('--diff_reg', default=0, type=int, help="Normalize regularization based on items given, smaller for more information. Higher for less.")
        my_parser.add_argument('--lr_decay', default=0, type=int, help="Print debugging messages")

        # -----------------------------
        # For debugging:
        # -----------------------------
        my_parser.add_argument('--debug_reg_z', default=0, type=int, help="|1|: Add additional information exactly"
                                                                          "|0|: No debugging")

    else:

        # -----------------------------
        # Sort of fixed
        # -----------------------------
        my_parser.add_argument('--runs', default=10, type=int, help="Number of epochs, i.e. runs through the datas set")
        my_parser.add_argument('--iterations', default=2000, type=int, help="Number of iterations of SGD")
        my_parser.add_argument('--folds', default=5, type=float, help="folds in CV, or outer folds in NCV")
        my_parser.add_argument('--split', default='NCV_NLPerf', type=str, help="|CV|: Cross Validation, w/o hyperparameter optimization; "
                                                                        "|NCV|: Nested CV, used for hyperparameter optimization "
                                                                        "|NCV_lolo_source|: Leave one language out."
                                                                        "|NCV_lolo_target|: Leave one language out.")
        # -----------------------------
        # Different cases to analyse
        # -----------------------------
        my_parser.add_argument('--task', default="wiki_aaMF", type=str, help="If only MF wanted, set: task_MF"
                                                                           "If MF with context wanted, set: task_aaMF (based on attribute aware MF)"
                                                                           "Available tasks:"
                                                                           "wiki, bli, wiki, bli, tsfpos, tsfel, tsfmt, tsfparsing")
        my_parser.add_argument('--dim', default=1, type=int, help="Number of latent dimensions")
        my_parser.add_argument('--attribute', default="ctx", type=str, help="Attributes to use: choose si, ctx, csi")
        my_parser.add_argument('--context', default=19, type=int, help="Available context for wiki MT:"
                                                                       "|1-15|: each feature individually"
                                                                       "|16|: Uriel lang. features,"
                                                                       "|17|: BLEU scores,"
                                                                       "|18|: 6 most correlated features,"
                                                                       "|19|: 11 most correlated features,"
                                                                       "|20|: 4 features, uncorrelated with BLEU,"
                                                                       "|21|: All avaiable features (w/o BLEU)")

        # -----------------------------
        # hyperparam optimization for model selection:
        # -----------------------------
        my_parser.add_argument('--lr', default=[0.01], type=float, help="Learning rate alpha")
        my_parser.add_argument('--reg_h', default=[0.1], type=float, help="Regularization param target languages")
        my_parser.add_argument('--reg_w', default=[0.1], type=float, help="Regularization param source languages")
        my_parser.add_argument('--reg_x', default=[0.01], type=float, help="Regularization param attribute_source")
        my_parser.add_argument('--reg_y', default=[0.01], type=float, help="Regularization param attribute_language")
        my_parser.add_argument('--reg_z', default=[0.01], type=float, help="Regularization param context matrix")  #0.00001
        my_parser.add_argument('--reg_bias_s', default=[0.01], type=float, help="Regularization param bias source language")
        my_parser.add_argument('--reg_bias_t', default=[0.01], type=float, help="Regularization param bias target language")

        # -----------------------------
        # Some other things:
        # -----------------------------
        my_parser.add_argument('--savepngs', default=True, type=bool, help="Save plots as pngs")
        my_parser.add_argument('--doplots', default=False, type=bool, help="Do plots")
        my_parser.add_argument('--enable_log', default=1, type=int, help="Print debugging messages")
        my_parser.add_argument('--diff_reg', default=0, type=int, help="Normalize regularization based on items given, smaller for more information. Higher for less.")
        my_parser.add_argument('--lr_decay', default=0, type=int, help="Print debugging messages")

        # -----------------------------
        # For debugging:
        # -----------------------------
        my_parser.add_argument('--debug_reg_z', default=0, type=int, help="|1|: Add additional information exactly"
                                                                          "|0|: No debugging")

    args = my_parser.parse_args()

    return args