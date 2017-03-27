# -*- coding: utf-8 -*-                     #
# ========================================= #
# Coalescent Simulations Main               #
# author      : Che Yeol (Jayeol) Chun      #
# last update : 03/17/2017                  #
# ========================================= #

import time

# from models import
from utils.utils import *
import argparse
import sys
# from utils.visualization.plots import plot_accurate_thold

from models.models import Kingman, BolthausenSznitman
from models.structure import Sample

from utils.visualization.plots import display_tree, display_stats

# Descriptions
MODELS = [Kingman, BolthausenSznitman]
MODELS = [Kingman]
STATS  = ['Bottom Branch Length']

color_list  = ['magenta', 'cyan']
param_list  = ["sample_size", "mutation_rate"]

# Settings
# num_iter_each = 1000        # how many iterations for a given combination of a sample size and mutation rate?
# init_mutation_rate = 0.1    # what mutation rate to begin src from?
#
# sample_size_step = 3
# sample_size_begin = 25
# sample_size_end = 30
# sample_size_range = np.arange(sample_size_begin, sample_size_end, sample_size_step)

def main(trees=None, exp=False, verbose=False):
    print("*** Coalescent Simulations Running...")
    # if mode: # single model experiment
    #     mode = select_mode()


    if trees:

        mu = 0.05
        sample_size = 12
        iter_num = 1

        for model in MODELS:

            model = model(sample_size, mu)
            data = np.zeros((iter_num, len(STATS)))

            for i in range(iter_num):
                coalescent_list = np.array([Sample(i + 1) for i in range(sample_size)])
                root = model.coalesce(coalescent_list, (i, data))
                display_tree((root,))

            print(model.data)
            sys.exit(0)




    else: # comparative studies between models
        pass




    # Begin Simulation
    # accurate_threshold_bbl, refined_mu = iterate(num_iter_each, sample_size_range, init_mutation_rate,
    #                                              param_list, model_list, color_list, stat_list)
    #
    # plot_accurate_thold(sample_size_range, accurate_threshold_bbl, refined_mu,
    #                     param_list, model_list, color_list, stat_list)
    #
    # print("\n*** Program Execution Time :", time.process_time(), "s ***\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", "-M", action="store_true",
    #                     help="option to specify coalescent model. DO NOT include "
    #                          "for comparative analysis between models")
    parser.add_argument("--trees", "-T", action="store_true",
                        help="create and plot trees for each model")
    # parser.add_argument("--e", "-E", action="store_true",
    #                     help="run experiments ; will be prompted for experimental settings and hyperparameter values")
    parser.add_argument("--verbose", "-V", action="store_true",
                        help="increase output verbosity")
    args = parser.parse_args()
    # main(args.m, args.e, args.v)
    main(trees=args.trees, verbose=args.verbose)