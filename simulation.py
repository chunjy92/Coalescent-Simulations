# -*- coding: utf-8 -*-                     #
# ========================================= #
# Coalescent Simulations Main               #
# author      : Che Yeol (Jayeol) Chun      #
# last update : 06/26/2016                  #
# ========================================= #

import numpy as np
import time
import coal_sims_def as cs_def

# Descriptions
model_list  = ['Kingman', 'Bolthausen-Sznitman']
color_list  = ['magenta', 'cyan']
stat_list   = ['Average Number of Mutations', 'Average Number of Ancestors', 'Bottom Branch Length', 'Threshold Branch Length']
m           = len(stat_list)  # Number of Features

param_list  = ["sample_size", "n", "mu", "num_desc_thold"]
cs_def.model_list, cs_def.color_list, cs_def.stat_list, cs_def.param_list = model_list, color_list, stat_list, param_list

# Default Test Simulation Parameters
default_val = (30, 500, 1.75, 0.80)

# For Testing Algorithm -- Uncomment the second line for use
test_parameter = (10, 5, 1.75, 0.5)
test_parameter = 0

if test_parameter:  sample_size, n, mu, num_desc_thold = test_parameter
else:               sample_size, n, mu, num_desc_thold = cs_def.set_parameters(default_val)

cs_def.sample_size, cs_def.n, cs_def.mu, cs_def.m, cs_def.num_desc_thold = sample_size, n, mu, m, num_desc_thold
cs_def.display_init_params()

# Default Scikit Stat Analysis Parameters
split_test_size   = 0.50
classifier_kernel = 'linear'

# Instructions
draw        = False
stat_show   = True
if n <= 10  : draw, stat_show = True, False

# Data Arrays
k_list, b_list = np.zeros((n, m)), np.zeros((n, m))

################################ Begin Simulation ################################

for i in range(n):
    # Set-Up
    kingman_coalescent_list = np.empty(sample_size, dtype=np.object)
    bs_coalescent_list = np.empty(sample_size, dtype=np.object)

    # Populate the lists with samples
    cs_def.populate_coalescent_list(kingman_coalescent_list, bs_coalescent_list)

    # Kingman Coalescent Model Tree Construction
    kingman_ancestor = cs_def.kingman_coalescence(kingman_coalescent_list, *(k_list, i))

    # Bolthausen-Sznitman Coalescent Model Tree Construction
    bs_ancestor = cs_def.bs_coalescence(bs_coalescent_list, *(b_list, i))

    # Only if the option to display the tree has been enabled
    if draw:    cs_def.display_tree((kingman_ancestor, bs_ancestor))

# Data Analysis & Display
data_k, data_b = np.transpose(k_list), np.transpose(b_list)
cs_def.display_stats(data_k, data_b, model_list, stat_list)

# Plot histogram of each data
cs_def.plot_histogram_each_data(data_k, data_b)

################################# End Simulation #################################

if stat_show: ############################### Begin Stat Analysis ###############################

    # Preprocessing data by splitting accordingly and scaling it
    X, y, X_train_raw, X_test_raw, y_train, y_test = cs_def.preprocess_data(k_list, b_list, test_size=split_test_size)
    X_train_scaled, X_test_scaled                  = cs_def.scale_X(X_train_raw, X_test_raw, y_train)

    # Define Classifier, train it and get its properties
    SVC_clf, coef, intercept = cs_def.define_classifier(X_train_scaled, y_train, kernel=classifier_kernel)

    # Get distances of each vector from the separating hyperplane
    SVC_dec, k_dec, b_dec = cs_def.get_decision_function(SVC_clf, X_test_scaled, y_test)

    # Accuracy Test
    cs_def.test_accuracy(SVC_clf, X_train_scaled, y_train, X_test_scaled, y_test)

    # Histogram
    cs_def.plot_SVC_decision_function_histogram(SVC_dec, k_dec, b_dec)

    # ROC Curve
    cs_def.plot_ROC_curve(X_train_scaled, X_test_scaled, y_train, y_test)

    # PCA
    cs_def.perform_pca(SVC_dec, X_test_raw, y_test, coef, three_d=True)

print("\n*** Program Execution Time :",time.process_time(),"s ***\n")
