# -*- coding: utf-8 -*-                     #
# ========================================= #
# Experiment Simulator                      #
# author      : Che Yeol (Jayeol) Chun      #
# last update : 04/08/2017                  #
# ========================================= #

import numpy as np
from models.structure import Sample
from utils.display import *

__author__ = 'Jayeol Chun'

def experiment(sample_size, num_test, mu, mu_step, models, num_iter, verbose=False):
    '''
    single experiment handled by one process
    i.e. executes $simulate method $test_range number of times
    and stores the summary statistics for each simulation: avg and variance.
    -- mix and match various combinations of sample size and mutation rate
    '''
    init_mu = mu
    data = [dict() for _ in range(len(models))]
    for test_index in range(num_test):
        print("\n***** {}th Test Iteration out of {} *****".format(test_index + 1, num_test), end='')
        sim_data = [np.zeros((num_iter, 1)) for _ in range(len(models))]
        mu = init_mu
        while mu < 2.0:  # 2.0 arbitrarily set for now
            display_params((sample_size, mu, num_iter))
            res = simulate(models, num_iter, sample_size, mu,
                           sim_data, exp=True, verbose=verbose)
            mu += mu_step
            for i in range(len(models)):
                key = "{} {:.2f}".format(sample_size, mu)
                if key in data[i]: data[i][key].append(res[i])
                else: data[i][key] = [res[i]]
    return data


def simulate(models, num_iter, sample_size, mu, data, exp=False, verbose=False):
    '''
    single simulation
    i.e. produces $num_iter many trees for each model
    and stores each statistics in to $data
    Returns some statistics, may change in the future
    '''
    for i, model in enumerate(models):
        model = model(sample_size, mu)
        for iter in range(num_iter):
            coalescent_list = [Sample(s + 1) for s in range(sample_size)]
            root = model.coalesce(coalescent_list, (iter, data[i]),
                                  exp=exp, verbose=verbose)
            if num_iter < 5: display_tree(root, verbose=verbose)

    return [(np.average(l), np.var(l)) for l in data]
