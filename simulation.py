# -*- coding: utf-8 -*-                     #
# ========================================= #
# Coalescent Simulations Main               #
# author      : Che Yeol (Jayeol) Chun      #
# last update : 06/29/2016                  #
# ========================================= #

import time
import coal_sims_def as cs_def

# Descriptions
model_list  = ['Kingman', 'Bolthausen-Sznitman']
color_list  = ['magenta', 'cyan']
stat_list   = ['Average Number of Mutations', 'Bottom Branch Length']
num_feature = len(stat_list)  # Number of Features
param_list  = ["sample_size", "n", "mu"]
cs_def.model_list, cs_def.color_list, cs_def.stat_list, cs_def.param_list = model_list, color_list, stat_list, param_list

# Default Test Simulation Parameters
default_val = (30, 500, 1.75)  # in param_list order

# For Testing -- Uncomment the second line for use
test_parameter = (300, 300, .7)
#test_parameter = (10, 3, .7)
test_parameter = 0

if test_parameter:
    sample_size, n, mu = test_parameter
else:
    sample_size, n, mu = cs_def.set_parameters(default_val)

# Initialize the Variables in the Definitions Module
cs_def.sample_size, cs_def.n, cs_def.mu, cs_def.num_feature = sample_size, n, mu, num_feature
cs_def.display_init_params()

# Default Sci-kit Stat Analysis Parameters
split_test_size   = 0.25
classifier_kernel = 'linear'

# Instructions
newick         = False  # whether to print the trees in newick format
plot_each_data = True  # whether to plot histogram for each feature data
ml             = True  # whether to perform machine learning on the data
pca_three_d    = False  # whether to print 3-d version of PCA

if n <= 10:
    newick, ml = True, False

########################################################################################################################

# Simulation
k_list, b_list = cs_def.simulation(newick=newick, plot_data=plot_each_data)

# Machine Learning
if ml:
    cs_def.perform_ml(k_list, b_list, test_size=split_test_size, classifier_kernel=classifier_kernel, pca_three_d=pca_three_d)

print("\n*** Program Execution Time :",time.process_time(),"s ***\n")
