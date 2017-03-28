# -*- coding: utf-8 -*-                     #
# ========================================= #
# Coalescent Simulations Main               #
# author      : Che Yeol (Jayeol) Chun      #
# last update : 03/27/2017                  #
# ========================================= #

import argparse
import time

from models.models import *
from utils.display import *
from utils.utils import *

__author__ = 'Jayeol Chun'

# Descriptions
MODELS = [Kingman, BolthausenSznitman]
STATS  = ['Bottom Branch Length']

# color_list  = ['magenta', 'cyan']
PARAMS = ["sample_size", "mutation_rate", "num_iter"]

def main(test=False, exp=False, verbose=False):
    print("*** Coalescent Simulations Running ***")
    if test:
        print("*** Test Mode ***")
        if verbose: print("--- Verbosity Increased")

        default = (10, 0.9, 1)

        # default params
        sample_size = 10
        mu = 0.9
        iter_num = 1

        # request user input
        # sample_size, mu, iter_num = request_user_input(PARAMS, default)

        display_params((sample_size, mu, iter_num))

        for model in MODELS:
            model = model(sample_size, mu)
            data = np.zeros((iter_num, len(STATS)))

            for i in range(iter_num):
                coalescent_list = [Sample(i+1) for i in range(sample_size)]
                root = model.coalesce(coalescent_list, (i, data), verbose=verbose)
                display_tree(root, verbose=verbose)

    else: # comparative studies between models
        pass

    print("*** Program Execution Time: {} s ***".format(time.process_time()))
    print("\n*** Coalescent Simulations Complete ***")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", "-M", action="store_true",
    #                     help="option to specify coalescent model. DO NOT include "
    #                          "for comparative analysis between models")
    parser.add_argument("--test", "-t", action="store_true",
                        help="test by creating and plotting trees for each model")
    # parser.add_argument("--e", "-E", action="store_true",
    #                     help="run experiments ; will be prompted for experimental settings and hyperparameter values")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="increase output verbosity")
    args = parser.parse_args()
    # main(args.m, args.e, args.v)
    main(test=args.test, verbose=args.verbose)