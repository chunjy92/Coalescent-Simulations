# -*- coding: utf-8 -*-                     #
# ========================================= #
# Coalescent Simulations Main               #
# author      : Che Yeol (Jayeol) Chun      #
# last update : 03/29/2017                  #
# ========================================= #

# TODO:
# 1) Implement Experiment mode asap
# 2) Implement SVM analysis + subsequent plots into test mode : done
# 3) Unittests
# 4) Multiprocessing

import argparse
import time
from models.models import *
from utils.display import *
from simulate import run_exeriment

__author__ = 'Jayeol Chun'

# Descriptions
MODELS = [Kingman, BolthausenSznitman]
STATS  = ['Bottom Branch Length']

def main(sample_size, sample_size_end, sample_size_step,
         mu, mu_step, iter_num, num_proc,
         test=False, verbose=False):

    print("*** Coalescent Simulations Running ***")
    if verbose: print("--- Verbosity Increased")
    if test:
        print("*** Test Mode ***")

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

        if sample_size_end <= sample_size:
            sample_size_end = sample_size + (sample_size_step * 3)+1 # arbitrary, run 3 different choices

        sample_size_range = range(sample_size, sample_size_end, sample_size_step)

        if num_proc > 1:
            print("\n*** MultiProcessing with {} Processes ***".format(num_proc))

        else:
            for model in MODELS:
                run_exeriment(model, sample_size_range, mu, mu_step, iter_num)
                pass
                # model = model
            pass

    print("*** Program Execution Time: {} s ***".format(time.process_time()))
    print("\n*** Coalescent Simulations Complete ***")


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

    parser.add_argument("-t", "--test", action="store_true",
                        help="test by creating and plotting trees for each model")
    parser.add_argument( "-v", "--verbose", action="store_true",
                        help="increase output verbosity")

    args = parser.parse_args()

    main(sample_size=args.sample_size, sample_size_end=args.sample_size_end, sample_size_step=args.sample_size_step,
         mu=args.mu, mu_step=args.mu_step, iter_num=args.num_iter,
         num_proc=args.num_proc, test=args.test,
         verbose=args.verbose)