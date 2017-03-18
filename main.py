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

# Descriptions
model_list  = ['Kingman', 'Bolthausen-Sznitman']
color_list  = ['magenta', 'cyan']
param_list  = ["sample_size", "mutation_rate"]
stat_list   = ['Bottom Branch Length']

# Settings
# num_iter_each = 1000        # how many iterations for a given combination of a sample size and mutation rate?
# init_mutation_rate = 0.1    # what mutation rate to begin src from?
#
# sample_size_step = 3
# sample_size_begin = 25
# sample_size_end = 30
# sample_size_range = np.arange(sample_size_begin, sample_size_end, sample_size_step)

def main(mode=None, exp=False, verbose=False):
    print("*** Coalescent Simulations Running...")
    if mode: # single model experiment
        mode = select_mode()


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
    parser.add_argument("--m", "-M", action="store_true",
                        help="option to specify coalescent model. DO NOT include "
                             "for comparative analysis between models")
    # parser.add_argument("--e", "-E", action="store_true",
    #                     help="run experiments ; will be prompted for experimental settings and hyperparameter values")
    parser.add_argument("--v", "-V", action="store_true",
                        help="increase output verbosity")
    args = parser.parse_args()
    # main(args.m, args.e, args.v)
    main(args.m, verbose=args.v)