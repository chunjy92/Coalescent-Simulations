# -*- coding: utf-8 -*-                     #
# ========================================= #
# Experiment Simulator                      #
# author      : Che Yeol (Jayeol) Chun      #
# last update : 03/28/2017                  #
# ========================================= #

__author__ = 'Jayeol Chun'

import numpy as np
from models.structure import Sample
from utils.display import display_params

def run_exeriment(model, sample_size_range,
                  mu=0.1, mu_step=0.02, iter_num=100, verbose=False):
    for sample_size in sample_size_range:
        while mu < 2: # 2 arbitrarily chosen
            display_params((sample_size, mu, iter_num))
            model = model(sample_size, mu)

            # need more comprehensive data..
            data = np.zeros((iter_num, len(STATS)))

            for i in range(iter_num):
                coalescent_list = [Sample(i+1) for i in range(sample_size)]
                model.coalesce(coalescent_list, (i, data), verbose=verbose)


            mu += mu_step
