# -*- coding: utf-8 -*-

import os
import sys
import time
import pickle
# import argparse
from argparse import ArgumentParser
from random import shuffle
from functools import partial
from multiprocessing import Pool
import numpy as np
from models import MODELS, Kingman, BolthausenSznitman
from tree_utils import load_trees, generate_trees
from simulation import simulate, experiment, analyze, display_params,  display_stats


__author__ = 'Jayeol Chun'

STATS  = ['Bottom Branch Length']
TREE = ".tree"


def test(sample_size: int, mu: float, num_iter: int, graphics=False, verbose=False):

    display_params((sample_size, mu, num_iter))
    data = [np.zeros((num_iter, len(STATS))) for _ in range(len(MODELS))]

    # single simulation
    simulate(MODELS, num_iter, sample_size, mu, data, exp=False, graphics=graphics, verbose=verbose)

    display_stats(data)
    if num_iter >= 10: analyze(data, graphics=graphics)


def etc(sample_size: int, sample_size_end: int, sample_size_step: int, mu: float, mu_thold: float, num_iter: int,
         num_proc: int, num_test: int, tree_dir: str=None, test: bool=False, graphics: bool=False, verbose: bool=False):
    '''
    main simulation/experiment script
    experiment mode by default
    '''


    # comparative studies between models
    print("Experiment Mode")

    if sample_size_end <= sample_size:
        sample_size_end = sample_size + (sample_size_step*3)+1 # arbitrary, run 3 different choices
    sample_sizes = list(range(sample_size, sample_size_end, sample_size_step))

    if num_proc > 1:
        print("\n******* Running Experiment with {} Processes *******".format(num_proc))


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
        print("\n******* Starting a Single Process Experiment *******")
        # data = [dict() for _ in range(len(MODELS))]
        data = dict()
        for sample_size in sample_sizes:
            res = experiment(sample_size, num_test, mu, mu_thold, MODELS, num_iter,
                             graphics=graphics, verbose=verbose)
            # for d, t in zip(data, res): d.update(t)
            data.update(res)
        # print(data)
        print(data)


def main():
    parser = ArgumentParser()

    population = parser.add_argument_group('Population Variables')
    population.add_argument("-n", "--sample_size", nargs='?', default=15, type=int,
                            help="init population sample size, subject to change in experiment. "
                             "The value is final for testing")
    population.add_argument("-e", "--sample_size_end", nargs='?', default=25, type=int,
                            help="end of sample size range for experiment")
    population.add_argument("-s", "--sample_size_step", nargs='?', default=3, type=int,
                            help="sample size range step unit for experiment")
    population.add_argument("-m", "--mu", nargs='?', default=0.5, type=float,
                            help="init mutation rate value, subject to change in experiment. "
                             "The value is final for testing")
    population.add_argument("-o", "--mu_th", nargs='?', default=5.0, type=float,
                            help="mu range step unit for experiment") # arbitrary..

    sims = parser.add_argument_group('Simulation & Experiment Variables')
    sims.add_argument("-i", "--num_iter", nargs='?', default=3, type=int,
                      help="number of iterations for one experiment or test")
    sims.add_argument("-p", "--num_proc", nargs='?', default=1, type=int,
                      help="number of processes for experiment (only for experiment mode)")
    sims.add_argument("-t", "--num_tests", nargs='?', default=1, type=int,
                      help="number of processes for experiment (only for experiment mode)")

    dirs = parser.add_argument_group('Directories')
    dirs.add_argument("--input_dir", type=str, default=None,
                      help="path to the saved trees")
    dirs.add_argument("--output_dir", type=str,default=None,
                      help="path to the trees to be saved")

    flags = parser.add_argument_group('Miscellaneous Flags for Experiment Control')
    flags.add_argument("--test", action="store_true",
                       help="test by creating and plotting trees for each model")
    flags.add_argument("--graphics", action="store_true",
                       help="produce plots or graphics. Default is no graphics.")
    flags.add_argument("--verbose", action="store_true",
                       help="increase output verbosity")

    args, unparsed = parser.parse_known_args()

    tic = time.time()
    print("******* Coalescent Simulations Main *******")

    if args.test:
        print("******* Testing Mode *******")
        generate_trees([args.sample_size], args.num_iter, output_dir=args.output_dir)
        # based on time trees, do more simulation
    elif args.input_dir:
        load_trees(args.input_dir)
    else:
        if args.sample_size_end <= args.sample_size:
            args.sample_size_end = args.sample_size + (args.sample_size_step*3) + 1  # arbitrary, run 3 different choices
        sample_sizes = list(range(args.sample_size, args.sample_size_end, args.sample_size_step))
        # print("Sample Sizes: {} to {}".format(args.sample_size, args.sample_size_end))
        # sys.exit(0)
        generate_trees(sample_sizes, args.num_iter, num_proc=args.num_proc,
                                    output_dir=args.output_dir)

    # At this point, we have time trees. Now apply mutation...



    print("\n******* Program Execution Time: {:.2f} s *******".format(time.time()-tic))
    print("******* Coalescent Simulations Complete *******")


if __name__ == '__main__':
    main()