# -*- coding: utf-8 -*-                     #
# ========================================= #
# Coalescent Simulations Main               #
# author      : Che Yeol (Jayeol) Chun      #
# last update : 04/01/2017                  #
# ========================================= #

# TODO:
# 1) Implement Experiment mode asap : done
# 2) Implement SVM analysis + subsequent plots into test mode : done, except for PCA (even necessary?)...
# 3) Unittests
# 4) Multiprocessing

import argparse
import time
from models.models import *
from utils.display import *
from utils.plot import *
from analyze import *
from simulate import *

__author__ = 'Jayeol Chun'

# Descriptions
MODELS = [Kingman, BolthausenSznitman]
STATS  = ['Bottom Branch Length']

def main(sample_size, sample_size_end, sample_size_step,
         mu, mu_step, iter_num, num_proc,
         test=False, verbose=False):
    '''
    experiment mode by default
    '''

    print("*** Coalescent Simulations Begin ***")
    if verbose: print("--- Verbosity Increased")
    if test:
        print("*** Test Mode ***")
        display_params((sample_size, mu, iter_num))

        # numpy data storage array
        data = [np.zeros((iter_num, len(STATS))) for _ in range(len(MODELS))]

        for i, model in enumerate(MODELS):
            model = model(sample_size, mu)

            for iter in range(iter_num):
                coalescent_list = [Sample(s+1) for s in range(sample_size)]
                root = model.coalesce(coalescent_list, (iter, data[i]), verbose=verbose)
                if iter_num < 5:
                    display_tree(root, verbose=verbose)

        print("\n... Test iterations done.")
        # display_stats()

        if iter_num >= 10:
            # Stat Analysis
            print("\n*** Analyzing Statistics ***\n")

            # data split by Kingmand and BS cateogry
            k_data, b_data = data
            X, y, splits = preprocess(k_data, b_data) # splits: train, test for each

            X_train, X_test = scale_X(*splits)
            y_train, y_test = splits[2], splits[3]

            # Initialize SVC
            clf, clf_dec = define_classifier(X_train, X_test, y_train)
            print("... Done.")

            # No need to test on train...
            # test_accuracy(clf, X_test, y_test)
            y_pred = clf.predict(X_test)
            print("\nTest Set Accuracy  :", metrics.accuracy_score(y_test, y_pred))

            print("\nPlotting Decision Function Histogram...")
            plot_SVC_decision_function_histogram(clf_dec, clf_dec[y_test==0], clf_dec[y_test==1])
            print("... Done.")

            print("Plotting ROC Curve...")
            plot_ROC_curve(X_train, X_test, y_train, y_test)
            print("... Done.")

            # perform_pca(clf_dec, X_test, y_test, clf.coef_[0], MODELS, three_d=True)


    else:
        # comparative studies between models
        print("*** Experiment Mode ***")

        if sample_size_end <= sample_size:
            sample_size_end = sample_size + (sample_size_step * 3)+1 # arbitrary, run 3 different choices

        sample_size_range = range(sample_size, sample_size_end, sample_size_step)

        if num_proc > 1:
            print("\n*** MultiProcessing with {} Processes ***".format(num_proc))

        else:
            data = [np.zeros((iter_num, len(STATS))) for _ in range(len(MODELS))]
            for i, model in enumerate(MODELS):
                run_exeriment(model, sample_size_range, mu, mu_step, iter_num)
                pass
                # model = model
            pass

    print("\n*** Program Execution Time: {} s ***".format(time.process_time()))
    print("*** Coalescent Simulations Complete ***")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--sample_size", nargs='?', default=5, type=int,
                        help="init population sample size, subject to change in experiment. The value is final for testing")
    parser.add_argument("-e", "--sample_size_end", nargs='?', default=15, type=int,
                        help="end of sample size range for experiment")
    parser.add_argument("-s", "--sample_size_step", nargs='?', default=5, type=int,
                        help="sample size range step unit for experiment")

    parser.add_argument("-m", "--mu", nargs='?', default=0.9, type=float,
                        help="init mutation rate value, subject to change in experiment. The value is final for testing")
    parser.add_argument("-o", "--mu_step", nargs='?', default=0.04, type=float,
                        help="mu range step unit for experiment")
    parser.add_argument("-i", "--num_iter", nargs='?', default=1, type=int,
                        help="number of iterations for one experiment or test")

    parser.add_argument("-p", "--num_proc", nargs='?', default=1, type=int,
                        help="number of processes for experiment (only for experiment mode)")

    parser.add_argument("--test", action="store_true",
                        help="test by creating and plotting trees for each model")
    # parser.add_argument("--no_graphics", action="store_true",
    #                     help="do not show any graphics")
    parser.add_argument("--verbose", action="store_true",
                        help="increase output verbosity")

    args = parser.parse_args()

    main(sample_size=args.sample_size, sample_size_end=args.sample_size_end, sample_size_step=args.sample_size_step,
         mu=args.mu, mu_step=args.mu_step, iter_num=args.num_iter, num_proc=args.num_proc,
         test=args.test, verbose=args.verbose)