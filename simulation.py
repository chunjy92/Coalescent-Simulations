# -*- coding: utf-8 -*-                     #
# ========================================= #
# Coalescent Simulations Main               #
# author      : Che Yeol (Jayeol) Chun      #
# last update : 09/03/2016                  #
# ========================================= #

import time
from coal_sims_def import *

# Descriptions
model_list  = ['Kingman', 'Bolthausen-Sznitman']
color_list  = ['magenta', 'cyan']
param_list  = ["sample_size", "mutation_rate"]
stat_list   = ['Bottom Branch Length']

# Settings
num_iter_each = 1000        # how many iterations for a given combination of a sample size and mutation rate?
init_mutation_rate = 0.1    # what mutation rate to begin simulation from?

sample_size_step = 3
sample_size_begin = 25
sample_size_end = 30
sample_size_range = np.arange(sample_size_begin, sample_size_end, sample_size_step)

# Begin Simulation
accurate_threshold_bbl, refined_mu = iterate(num_iter_each, sample_size_range, init_mutation_rate,
                                             param_list, model_list, color_list, stat_list)

plot_accurate_thold(sample_size_range, accurate_threshold_bbl, refined_mu,
                    param_list, model_list, color_list, stat_list)

print("\n*** Program Execution Time :", time.process_time(), "s ***\n")
