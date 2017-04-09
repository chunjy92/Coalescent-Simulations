# -*- coding: utf-8 -*-                     #
# ========================================= #
# Coalescent Simulations Main               #
# author      : Che Yeol (Jayeol) Chun      #
# last update : 04/08/2017                  #
# ========================================= #

# TODO:
# 1. upload to AWS
# 2. scale up experiments, perform various tests
# should begin saving the output stats

import time
import argparse
from functools import partial
from multiprocessing import Pool
import numpy as np

from models.models import Kingman, BolthausenSznitman
from analyze import analyze
from simulate import *

__author__ = 'Jayeol Chun'

# Descriptions
MODELS = [Kingman, BolthausenSznitman]
STATS  = ['Bottom Branch Length']


def main(sample_size, sample_size_end, sample_size_step, mu, mu_step,
         num_iter, num_proc, num_test, test=False, verbose=False):
    '''
    experiment mode by default
    '''

    tic = time.time()
    print("\n*** Coalescent Simulations Main ***")
    if test:
        print("*** Testing Mode ***")
        display_params((sample_size, mu, num_iter))

        # numpy data storage array
        data = [np.zeros((num_iter, len(STATS))) for _ in range(len(MODELS))]

        # single simulation
        res = simulate(MODELS, num_iter, sample_size, mu, data, exp=False, verbose=verbose)
        print("Test Simulation Done.")
        display_stats(res, STATS)
        if num_iter >= 10: analyze(data)

    else:
        # comparative studies between models
        print("*** Experiment Mode ***")
        if sample_size_end <= sample_size:
            sample_size_end = sample_size + (sample_size_step * 3)+1 # arbitrary, run 3 different choices
        sample_sizes = range(sample_size, sample_size_end, sample_size_step)

        if num_proc > 1:
            print("\n*** Running Experiment with {} Processes ***".format(num_proc))
            expr_wrapper = partial(experiment, num_test=num_test, mu=mu, mu_step=mu_step,
                           models=MODELS, num_iter=num_iter, verbose=False) # verbosity will only be confusing
            with Pool(processes=num_proc) as pool:
                res = pool.map_async(expr_wrapper, sample_sizes)
                pool.close()
                pool.join()

            # $data has all the data in dictionary format
            # key = (sample_size mu)
            # val = list of statistics from each simulation
            data = res.get()
            print("Done.")
            # print(data)

            # do sth with data...

        else:
            print("\n*** Starting a Single Process Experiment ***")
            data = [dict() for _ in range(len(MODELS))]
            for sample_size in sample_sizes:
                tmp = experiment(sample_size, num_test, mu, mu_step, MODELS, num_iter, verbose=verbose)
                for d, t in zip(data, tmp): d.update(t)
            print("Done.")
            print(data)

    print("\n*** Program Execution Time: {:.2f} s ***".format(time.time()-tic))
    print("*** Coalescent Simulations Complete ***")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--sample_size", nargs='?', default=5, type=int,
                        help="init population sample size, subject to change in experiment. "
                             "The value is final for testing")
    parser.add_argument("-e", "--sample_size_end", nargs='?', default=15, type=int,
                        help="end of sample size range for experiment")
    parser.add_argument("-s", "--sample_size_step", nargs='?', default=5, type=int,
                        help="sample size range step unit for experiment")

    parser.add_argument("-m", "--mu", nargs='?', default=0.9, type=float,
                        help="init mutation rate value, subject to change in experiment. "
                             "The value is final for testing")
    parser.add_argument("-o", "--mu_step", nargs='?', default=0.5, type=float,
                        help="mu range step unit for experiment")
    parser.add_argument("-i", "--num_iter", nargs='?', default=300, type=int,
                        help="number of iterations for one experiment or test")

    parser.add_argument("-p", "--num_proc", nargs='?', default=1, type=int,
                        help="number of processes for experiment (only for experiment mode)")
    parser.add_argument("-t", "--num_tests", nargs='?', default=3, type=int,
                        help="number of processes for experiment (only for experiment mode)")

    parser.add_argument("--test", action="store_true",
                        help="test by creating and plotting trees for each model")
    parser.add_argument("--verbose", action="store_true",
                        help="increase output verbosity")

    args = parser.parse_args()

    main(sample_size=args.sample_size, sample_size_end=args.sample_size_end, sample_size_step=args.sample_size_step,
         mu=args.mu, mu_step=args.mu_step, num_iter=args.num_iter, num_proc=args.num_proc, num_test=args.num_tests,
         test=args.test, verbose=args.verbose)