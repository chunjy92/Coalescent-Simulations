# -*- coding: utf-8 -*-                     #
# ========================================= #
# Coalescent Simulations Main               #
# author      : Che Yeol (Jayeol) Chun      #
# last update : 06/21/2016                  #
# ========================================= #

import numpy as np
import coal_sims_def as cs_def
import time

# Descriptions
model_list  = ['Kingman', 'Bolthausen-Sznitman']
color_list  = ['magenta', 'cyan']
stat_list   = ['Average Number of Mutations', 'Average Number of Ancestors', 'Mean Separation Time', 'Average Height',
              'Bottom Branch Length', 'Measure of Asymmetry using Variance Only', 'Most frequently Observed Mutation Value']
cs_def.model_list, cs_def.color_list, cs_def.stat_list = model_list, color_list, stat_list

# Simulation Parameters
sample_size = 30
n           = 500
mu          = 1.95
m           = len(stat_list)  # Number of Parameters(Features)
cs_def.sample_size, cs_def.n, cs_def.mu, cs_def.m = sample_size, n, mu, m

# Scikit Stat Analysis Parameters
classifier_kernel = 'linear'
split_test_size   = 0.50
pca_n_comp        = 2
hist_lin_space    = 30

# Data Arrays
k_list, b_list = np.zeros((n, m)), np.zeros((n, m))

# Instructions
draw        = False
stat_show   = True
if n <= 10  : draw, stat_show = True, False
if n > 1000 : cs_def.check_n()  # warning : n might have been set unnecessarily big for a test-case

################################ Begin Simulation ################################

for i in range(n):
    # Set-Up
    kingman_coalescent_list = np.empty(sample_size, dtype=np.object)
    bs_coalescent_list = np.empty(sample_size, dtype=np.object)

    # Populate the lists with samples
    cs_def.populate_coalescent_list(kingman_coalescent_list, bs_coalescent_list)

    # Kingman Coalescent Model Tree Construction
    kingman_coalescent_list = cs_def.kingman_coalescence(kingman_coalescent_list, *(k_list, i))

    # Bolthausen-Sznitman Coalescent Model Tree Construction
    bs_coalescent_list = cs_def.bs_coalescence(bs_coalescent_list, *(b_list, i))

    kingman_ancestor, bs_ancestor = kingman_coalescent_list[0], bs_coalescent_list[0]

    # Only if the option to display the tree has been enabled
    if draw:    cs_def.display_tree([kingman_ancestor, bs_ancestor])


# Data Analysis & Display
data_k, data_b = np.transpose(k_list), np.transpose(b_list)
cs_def.display_stats(data_k, data_b, model_list, stat_list)

# Plot histogram of each data
cs_def.plot_histogram_each_data(data_k, data_b, num_linspace=hist_lin_space)

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
    cs_def.perform_pca(X_train_raw, X_test_raw, y, coef, n_comp=pca_n_comp)

print("\n\nProgram Execution Time :",time.process_time(),"s")