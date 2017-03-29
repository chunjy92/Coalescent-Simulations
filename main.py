# -*- coding: utf-8 -*-                     #
# ========================================= #
# Coalescent Simulations Main               #
# author      : Che Yeol (Jayeol) Chun      #
# last update : 03/28/2017                  #
# ========================================= #

# TODO:
# 1) Implement Experiment mode asap
# 2) Implement SVM analysis + subsequent plots into test mode
# 3) Unittests
# 4) Multiprocessing

import argparse
import time
from models.models import *
from utils.display import *
from utils.utils import *
from simulate import run_exeriment

__author__ = 'Jayeol Chun'

# Descriptions
MODELS = [Kingman, BolthausenSznitman]
STATS  = ['Bottom Branch Length']

def main(num_proc=1, test=False, verbose=False):
    print("*** Coalescent Simulations Running ***")
    if verbose: print("--- Verbosity Increased")
    if test:
        print("*** Test Mode ***")
        # default params
        sample_size = 10
        mu = 0.9
        iter_num = 1

        display_params((sample_size, mu, iter_num))

        for model in MODELS:
            model = model(sample_size, mu)
            data = np.zeros((iter_num, len(STATS)))

            for i in range(iter_num):
                coalescent_list = [Sample(i+1) for i in range(sample_size)]
                root = model.coalesce(coalescent_list, (i, data), verbose=verbose)
                if iter_num < 5:
                    display_tree(root, verbose=verbose)
            # print(data)
    else: # comparative studies between models
        print("*** Experiment Mode ***")
        # default params
        sample_size = range(30, 10, 51)
        mu = 0.1
        mu_step = 0.02
        iter_num = 300

        if num_proc > 1:
            print("\n*** MultiProcessing with {} Processes ***".format(num_proc))

        else:
            for model in MODELS:
                run_exeriment(model, sample_size, mu, iter_num)
                pass
                # model = model
            pass


    print("*** Program Execution Time: {} s ***".format(time.process_time()))
    print("\n*** Coalescent Simulations Complete ***")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", "-M", action="store_true",
    #                     help="option to specify coalescent model. DO NOT include "
    #                          "for comparative analysis between models")
    parser.add_argument("-t", "--test", action="store_true",
                        help="test by creating and plotting trees for each model")
    parser.add_argument("-n", "--num_proc", nargs='?', default=1, type=int,
                        help="number of processes for experiment (only for experiment mode)")
    # parser.add_argument("--e", "-E", action="store_true",
    #                     help="run experiments ; will be prompted for experimental settings and hyperparameter values")
    parser.add_argument( "-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    args = parser.parse_args()

    main(num_proc=args.num_proc,
         test=args.test,
         verbose=args.verbose)