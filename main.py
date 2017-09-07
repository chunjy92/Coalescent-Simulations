# -*- coding: utf-8 -*-

import time
from functools import partial
from multiprocessing import Pool
from random import shuffle

import numpy as np

from models import MODELS
from simulation import simulate, experiment, analyze, display_params, display_stats
from tree_utils import load_trees, generate_trees
from config import get_config

__author__ = 'Jayeol Chun'

STATS  = ['Bottom Branch Length']


# def test(sample_size: int, mu: float, num_iter: int, graphics=False, verbose=False):
#
#   display_params((sample_size, mu, num_iter))
#   data = [np.zeros((num_iter, len(STATS))) for _ in range(len(MODELS))]
#
#   # single simulation
#   simulate(MODELS, num_iter, sample_size, mu, data, exp=False, graphics=graphics, verbose=verbose)
#
#   display_stats(data)
#   if num_iter >= 10: analyze(data, graphics=graphics)


# def etc(sample_size: int, sample_size_end: int, sample_size_step: int, mu: float, mu_thold: float, num_iter: int,
#          num_proc: int, num_test: int, tree_dir: str=None, test: bool=False, graphics: bool=False, verbose: bool=False):
#     '''
#     main simulation/experiment script
#     experiment mode by default
#     '''
#     # comparative studies between models
#     print("Experiment Mode")
#
#     if sample_size_end <= sample_size:
#         sample_size_end = sample_size + (sample_size_step*3)+1 # arbitrary, run 3 different choices
#     sample_sizes = list(range(sample_size, sample_size_end, sample_size_step))
#
#     if num_proc > 1:
#         print("\nRunning Experiment with {} Processes".format(num_proc))
#
#
#         shuffle(sample_sizes)
#         expr_wrapper = partial(experiment, num_test=num_test, mu=mu, mu_thold=mu_thold, models=MODELS,
#                                num_iter=num_iter, graphics=graphics, verbose=False)
#         with Pool(processes=num_proc) as pool:
#             res = pool.map_async(expr_wrapper, sample_sizes)
#             pool.close()
#             pool.join()
#
#         # $data has all the data in dictionary
#         # key = (sample_size mu) | val = list of statistics from each simulation
#         data = res.get()
#         print(data)
#
#     else:
#         print("\nStarting a Single Process Experiment")
#         # data = [dict() for _ in range(len(MODELS))]
#         data = dict()
#         for sample_size in sample_sizes:
#             res = experiment(sample_size, num_test, mu, mu_thold, MODELS, num_iter,
#                              graphics=graphics, verbose=verbose)
#             # for d, t in zip(data, res): d.update(t)
#             data.update(res)
#         # print(data)
#         print(data)


def main(config):
  tic = time.time()
  print("******* Coalescent Simulations Main *******")

  if config.test:
    print("Testing Mode")
    models = generate_trees([config.sample_size], config.num_iter, output_dir=config.output_dir)
  elif config.input_dir:
    models = load_trees(config.input_dir)
  else:
    if config.sample_size_end <= config.sample_size:
      config.sample_size_end = config.sample_size + (config.sample_size_step*3) + 1  # arbitrary, run 3 different choices
    sample_sizes = list(range(config.sample_size, config.sample_size_end, config.sample_size_step))
    models = generate_trees(sample_sizes, config.num_iter, num_proc=config.num_proc,
                            output_dir=config.output_dir, verbose=config.verbose)

  # At this point, we have time trees. Now apply mutation...

  print("\n******* Program Execution Time: {:.2f} s *******".format(time.time()-tic))

if __name__ == '__main__':
  main(get_config())
