# -*- coding: utf-8 -*-

# TODO:
# 1. upload to AWS
# 2. scale up experiments, perform various tests
# should begin saving the output stats

import argparse
import time
import numpy as np
from models import MODELS
from simulation import simulate, experiment, analyze

__author__ = 'Jayeol Chun'

STATS  = ['Bottom Branch Length']


def main(sample_size: int, sample_size_end: int, sample_size_step: int, mu: float, mu_thold: float,
         num_iter: int, num_proc: int, num_test: int, test: bool=False, graphics: bool=False, verbose: bool=False):
    '''
    main simulation/experiment script
    experiment mode by default
    '''

    tic = time.time()
    print("*** Coalescent Simulations Main ***")
    if test:
        print("*** Testing Mode ***")
        data = [np.zeros((num_iter, len(STATS))) for _ in range(len(MODELS))]

        # single simulation
        simulate(MODELS, num_iter, sample_size, mu, data, exp=False, graphics=graphics, verbose=verbose)
        if num_iter >= 10: analyze(data, graphics=graphics)

    else:
        # comparative studies between models
        print("*** Experiment Mode ***")

        if sample_size_end <= sample_size:
            sample_size_end = sample_size + (sample_size_step * 3)+1 # arbitrary, run 3 different choices
        sample_sizes = list(range(sample_size, sample_size_end, sample_size_step))

        if num_proc > 1:
            print("\n*** Running Experiment with {} Processes ***".format(num_proc))
            from random import shuffle
            from functools import partial
            from multiprocessing import Pool

            shuffle(sample_sizes)

            expr_wrapper = partial(experiment, num_test=num_test, mu=mu, mu_thold=mu_thold, models=MODELS,
                                   num_iter=num_iter, graphics=graphics, verbose=False)
            with Pool(processes=num_proc) as pool:
                res = pool.map_async(expr_wrapper, sample_sizes)
                pool.close()
                pool.join()

            # $data has all the data in dictionary
            # key = (sample_size mu) | val = list of statistics from each simulation
            data = res.get()
            print(data)

        else:
            print("\n*** Starting a Single Process Experiment ***")
            data = [dict() for _ in range(len(MODELS))]
            for sample_size in sample_sizes:
                tmp = experiment(sample_size, num_test, mu, mu_thold, MODELS, num_iter,
                                 graphics=graphics, verbose=verbose)
                for d, t in zip(data, tmp): d.update(t)

    print("Done.")
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
    parser.add_argument("-o", "--mu_threshold", dest="mu_thold", nargs='?', default=2.0, type=float,
                        help="mu range step unit for experiment")
    parser.add_argument("-i", "--num_iter", nargs='?', default=300, type=int,
                        help="number of iterations for one experiment or test")

    parser.add_argument("-p", "--num_proc", nargs='?', default=1, type=int,
                        help="number of processes for experiment (only for experiment mode)")
    parser.add_argument("-t", "--num_tests", nargs='?', default=3, type=int,
                        help="number of processes for experiment (only for experiment mode)")

    parser.add_argument("--test", action="store_true",
                        help="test by creating and plotting trees for each model")
    parser.add_argument("--graphics", action="store_true",
                        help="produce plots or graphics. Default is no graphics.")
    parser.add_argument("--verbose", action="store_true",
                        help="increase output verbosity")

    args = parser.parse_args()

    main(sample_size=args.sample_size, sample_size_end=args.sample_size_end, sample_size_step=args.sample_size_step,
         mu=args.mu, mu_thold=args.mu_thold, num_iter=args.num_iter, num_proc=args.num_proc, num_test=args.num_tests,
         test=args.test, graphics=args.graphics, verbose=args.verbose)
