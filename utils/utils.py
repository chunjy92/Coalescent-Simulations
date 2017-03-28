# -*- coding: utf-8 -*-                     #
# ========================================= #
# Coalescent Simulations Main Helper        #
# author      : Che Yeol (Jayeol) Chun      #
# last update : 03/27/2017                  #
# ========================================= #

import sys
import numpy as np
from typing import List

__author__ = 'Jayeol Chun'

def select_mode():
    '''
    # when -M or --m specified at command line
    allows user to select a single model
    '''
    print("\n*** Option to Specify Coalescent Model:")
    usr = num_trials = 0
    while usr == 0:
        usr = input("Select 1 for Kingman, 2 for Bolthausen-Sznitman:\n")
        try:
            usr = int(usr)
            if usr == 1:
                mode = "K"
                print("\n*** Kingman Coalescent Selected")
                break
            elif usr == 2:
                mode = "B"
                print("\n*** Bolthausen-Sznitman Coalescent Selected")
                break
            usr = 0
        except ValueError:
            pass
        num_trials += 1
        if num_trials >= 5:
            print("\n*** Fatal: Exceeded Max number of attempts\n*** System Exiting...")
            sys.exit(1)
    return mode

########################### Main Simulation ###########################

# def iterate(num_iter_each, sample_size_range, init_mu,
#             param_list, model_list, color_list, stat_list):
#     """
#     iterate through every combination of sample size and mutation rates given
#     """
#
#     accurate_threshold_bbl, refined_mu = np.zeros_like(sample_size_range, dtype=float), np.zeros_like(sample_size_range, dtype=float)
#     pass_percent = .9
#     for idx, sample_size in enumerate(sample_size_range):
#         mu = init_mu if init_mu < 15 else 200
#         while True:
#             __display_params(param_list, sample_size, mu)
#             threshold_bbl, percent_correct = simulation(sample_size, num_iter_each, mu,
#                                                         model_list, color_list, stat_list)
#             accurate_threshold_bbl[idx] = threshold_bbl
#             refined_mu[idx] = mu
#             if percent_correct >= pass_percent:
#                 print("\n*******Either equal to or over {:.2f}, moving along\n".format(pass_percent))
#                 break
#
#             mu = mu * (1 + pass_percent - percent_correct)
#             if mu >= 10000:
#                 break
#
#     return accurate_threshold_bbl, refined_mu

# def simulation(sample_size, niter, mu,
#                model_list, color_list, stat_list,
#                newick=False, plot_data=False):
#     """
#     simulates the Kingman and Bolthausen-Sznitman coalescence
#     @return : refer to return of get_threshold
#     """
#     num_feature = len(stat_list)
#     k_list, b_list = np.zeros((niter, num_feature)), np.zeros((niter, num_feature))
#     for i in range(niter):
#         # Populate the lists with samples
#         kingman_coalescent_list, bs_coalescent_list = __init_coalescent_list(sample_size)
#
#         # Kingman Coalescence
#         __kingman_coalescence(sample_size, mu,
#                               kingman_coalescent_list,
#                               *(k_list, i), newick=newick)
#
#         # Bolthausen-Sznitman Coalescence
#         __bs_coalescence(sample_size, mu,
#                          bs_coalescent_list,
#                          *(b_list, i), newick=newick)
#
#     return get_threshold(k_list, b_list)

# def get_threshold(k_list, b_list):
#     """
#     get BBL threshold
#     @param k_list: Kingman data
#     @param b_list: Bolthausen-Sznitman data
#     @return      : threhold values, correct prediction percentage
#     """
#     total_list = np.concatenate([k_list, b_list])
#     total_size = len(total_list)
#     k_label, b_label = np.zeros_like(k_list), np.ones_like(b_list)
#     total_label = np.concatenate([k_label, b_label])
#     result = np.concatenate([total_list, total_label], axis=1)
#     sorted_result = result[np.argsort(result[:, 0])]
#
#     threshold_bbl, threshold_idx = 0, 0
#     goodness, max_goodness = 0, 0
#     goodness_log = []
#     for index, res in enumerate(sorted_result):
#         i = res[1]
#         if i == 0:
#             goodness += 1
#         else:
#             goodness -= 1
#         goodness_log.append(goodness)
#         if goodness > max_goodness:
#             max_goodness = goodness
#             threshold_idx = index
#             try:
#                 threshold_bbl = (res[0] + sorted_result[index+1, 0]) / 2
#             except IndexError:
#                 threshold_bbl = res[0]
#
#     pred_label = np.concatenate([np.zeros(threshold_idx+1), np.ones(total_size - (threshold_idx+1))])
#     sum_correct = np.sum(pred_label == sorted_result[:, 1])
#     percent_correct = sum_correct / total_size
#     return threshold_bbl, percent_correct