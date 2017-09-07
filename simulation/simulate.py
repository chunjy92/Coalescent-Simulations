# -*- coding: utf-8 -*-

from typing import List

import numpy as np

from models import M, Sample
from .utils import display_params

__author__ = 'Jayeol Chun'


def experiment(sample_size: int, num_test: int, mu: float, mu_thold: float, models: M,
               num_iter: int, graphics: bool=False, verbose: bool=False):
  '''
  single experiment handled by one process
  i.e. executes $simulate method $test_range number of times
  and stores the summary statistics for each simulation: avg and variance.
  -- mix and match various combinations of sample size and mutation rate
  '''
  init_mu, mu_step = mu, 0.03
  # data = [dict() for _ in range(len(models))]
  data = dict()
  for test_index in range(num_test):
    print("\n******* {}th Test Iteration out of {} *******".format(test_index+1, num_test), end='')
    sim_data = [np.zeros((num_iter, 1)) for _ in range(len(models))]
    mu = init_mu
    while mu < mu_thold:  # 5.0 for now
      display_params((sample_size, mu, num_iter))
      res = simulate(models, num_iter, sample_size, mu, sim_data,
                     exp=True, graphics=graphics, verbose=verbose)

      # get threshold at 90 % precision
      bbl_thold, correct = get_threshold(sim_data, verbose=verbose)
      print("\nBottom Branch Length Threshold Value: {}".format(bbl_thold))
      print("Percentage Correct:                   {:.2f}".format(correct))

      # for i in range(len(models)):
      #     key = "{} {:.2f}".format(sample_size, mu)
      #     if key in data[i]: data[i][key].append(res[i])
      #     else: data[i][key] = [res[i]]

      if correct>=.90:
        print("Exiting")
        # for i in range(len(models)):
        key = str(sample_size)
        if key in data: data[key].append(bbl_thold)
        else: data[key] = [bbl_thold]
        print(data)
        break

      # optimize
      mu += mu_step

  return data


def simulate(sample_size, model, num_iter, verbose=False):
  # data = [np.zeros((num_iter, len(STATS))) for _ in range(len(MODELS))]
  # for sample_size in self.sample_sizes:
  #     key = str(sample_size)
  #     for iter in range(self.num_iter):
  #         coalescent_list = [Sample(s+1) for s in range(sample_size)]
  #         root = self._coalesce(coalescent_list, sample_size, verbose=verbose)
  #         self._store_tree(root, key)
  #         display_tree(root)

  # key = str(sample_size)
  # key = sample_si
  # print("Simjulating with", str(key))
  if verbose:
    print("\nSimjulating")
  time_trees = {}
  for _ in range(num_iter):
    coalescent_list = [Sample(s+1) for s in range(sample_size)]
    root = model.coalesce(coalescent_list, sample_size, verbose=verbose)
    # model.store_tree(root, key)
    if sample_size in time_trees:
      time_trees[sample_size].append(root)
    else:
      time_trees[sample_size] = [root]

  print("Ended with {}, root identity -> {}".format(str(sample_size), root.identity))
  return time_trees


def _simulate(models: M, num_iter: int, sample_size: int, mu: float, data: List[np.ndarray],
              tree_dir:str=None, exp: bool=False, graphics: bool=False, verbose: bool=False):
  '''
  single simulation
  i.e. produces $num_iter many trees for each model
  and stores each statistics in to $data
  Returns some statistics, may change in the future
  '''
  # a_path = "/Users/jayeolchun/Documents/Research/Coalescent-Simulations/data"
  for i, model in enumerate(models):
    model = model(sample_size, mu)
    for iter in range(num_iter):
      coalescent_list = [Sample(s+1) for s in range(sample_size)]
      root = model.coalesce(coalescent_list, (iter, data[i]), exp=exp, verbose=verbose)
      model.store_tree(root)
      # if num_iter < 5 and graphics: display_tree(root, verbose=verbose)
    print(model.time_trees)
    model.save_trees(tree_dir)
  res = [(np.average(l), np.var(l)) for l in data]

  return res



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




def get_threshold(data, verbose=False):
  total_list = np.concatenate(data)
  total_size = len(total_list)
  total_label = np.concatenate([np.zeros_like(data[0]),np.ones_like(data[1])])
  result = np.concatenate([total_list, total_label], axis=1)
  sorted_result = result[np.argsort(result[:,0])]

  threshold_bbl = threshold_idx = 0
  goodness = max_goodness = 0
  goodness_log = []
  for index, res in enumerate(sorted_result):
    i = res[1]
    goodness = goodness+1 if i==0 else goodness-1
    goodness_log.append(goodness)
    if goodness > max_goodness:
      max_goodness = goodness
      threshold_idx = index
      try:
        threshold_bbl = (res[0]+sorted_result[index+1, 0]) / 2
      except IndexError:
        threshold_bbl = res[0]

  pred_label = np.concatenate([np.zeros(threshold_idx+1), np.ones(total_size-(threshold_idx+1))])
  sum_correct = np.sum(pred_label==sorted_result[:,1])
  percent_correct = sum_correct / total_size
  return threshold_bbl, percent_correct
