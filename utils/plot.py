# -*- coding: utf-8 -*-                     #
# ========================================= #
# Coalescent Simulations Visualization      #
# author      : Che Yeol (Jayeol) Chun      #
# last update : 04/01/2017                  #
# ========================================= #

import numpy as np
from sklearn.linear_model.logistic import LogisticRegression
from sklearn import preprocessing, metrics, decomposition
import matplotlib.pyplot as plt

from utils.utils import project_onto_plane

__author__ = 'Jayeol Chun'

COLORS = ['magenta', 'cyan']

def goodness_vs_threshold(goodness, bbl):
    """
    plots the goodness vs. Bottom Branch Length Threshold
    @param goodness : 1-d Array - accumulation of correct predictions
    @param bbl      : 1-d Array - BBL Threshold
    """
    plt.figure()
    plt.plot(bbl, goodness, label='over time')
    plt.xlim([-1, 10])
    plt.title("Goodness vs BBL Threshold")
    plt.show()

def plot_accurate_thold(sample_size_range, accurate_threshold_bbl, refined_mu,
                        param_list, model_list, COLORS, stat_list):
    """
    plots various log-scaled versions of sample_size vs mu
    """
    for i in range(4):
        plt.figure()
        plt.scatter(sample_size_range, refined_mu, alpha=0.5, label="BBL")
        if i == 1:
            plt.xscale('log')
            plt.xlabel('LOG')
        elif i == 2:
            plt.yscale('log')
            plt.ylabel('LOG')
        elif i == 3:
            plt.xscale('log')
            plt.xlabel('LOG')
            plt.yscale('log')
            plt.ylabel('LOG')
        plt.show()

def plot_data(result_list, percent_list, sample_size_range, mutation_rate_range,
              model_list, COLORS):

    # Result List
    fig = plt.figure(figsize=(10, 8))
    ax = fig.gca(projection='3d')
    plt.rcParams['legend.fontsize'] = 10

    #for index in range(len(model_list)):
    for i in range(len(sample_size_range)):
        for j in range(len(mutation_rate_range)):
            xs = sample_size_range[i]
            ys = mutation_rate_range[j]
            zs = result_list[i][j]
            ax.scatter(xs, ys, zs, alpha=0.5,label="BBL" if i == 0 and j == 0 else "")


    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Mutation Rate')
    ax.set_zlabel('Bottom Branch Length Threshold Value')
    plt.title('BBL Scatter Plot')
    ax.legend(bbox_to_anchor=(0.35, 0.9))
    plt.show()

    # Percent_ilst
    fig = plt.figure(figsize=(10, 8))
    ax = fig.gca(projection='3d')
    plt.rcParams['legend.fontsize'] = 10

    # for index in range(len(model_list)):
    for i in range(len(sample_size_range)):
        for j in range(len(mutation_rate_range)):
            xs = sample_size_range[i]
            ys = mutation_rate_range[j]
            zs = percent_list[i][j]
            ax.scatter(xs, ys, zs, alpha=0.5, label="BBL" if i == 0 and j == 0 else "")

    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Mutation Rate')
    ax.set_zlabel('Percent Correct')
    plt.title('Percent Correct Scatter Plot')
    ax.legend(bbox_to_anchor=(0.35, 0.9))
    plt.show()

def plot_histogram_each_data(data_k, data_b):
    """
    plots histogram for each kind of data collected for each tree
    @param data_k : 2-d Array - holds kingman data
    @param data_b : 2-d Array - holds bolthausen data
    """
    plt.figure()
    num_linspace = 30
    stat_min, stat_max = np.amin(np.append(data_k, data_b)), np.amax(np.append(data_k, data_b))
    if np.absolute(stat_max-stat_min) >= 100:
        num_linspace += int(np.sqrt(np.absolute(stat_max-stat_min)))
    bins = np.linspace(np.ceil(stat_min)-1, np.ceil(stat_max)+1, num_linspace)
    plt.hist(data_k, facecolor='magenta', bins=bins, lw=1, alpha=0.5, label='Kingman')
    plt.hist(data_b, facecolor='cyan', bins=bins, lw=1, alpha=0.5, label='Bolthausen-Sznitman')
    plt.title('BBL')
    plt.legend(loc='upper right')
    plt.show()

def plot_histogram_each_data(data_k, data_b, num_linspace=30):
    """
    plots histogram for each kind of data collected for each tree
    @param data_k       : 2-d Array - holds kingman data
    @param data_b       : 2-d Array - holds bolthausen data
    @param num_linspace : Int       - defines how to space out the histogram bin
    """
    for i in range(m):
        plt.figure()
        stat_min, stat_max = np.amin(np.append(data_k[i], data_b[i])), np.amax(np.append(data_k[i], data_b[i]))
        if np.absolute(stat_max-stat_min) >= 100: num_linspace += int(np.sqrt(np.absolute(stat_max-stat_min)))
        bins = np.linspace(np.ceil(stat_min)-1, np.ceil(stat_max)+1, num_linspace)
        plt.hist(data_k[i], facecolor=COLORS[0], bins=bins, lw=1, alpha=0.5, label='Kingman')
        plt.hist(data_b[i], facecolor=COLORS[1], bins=bins, lw=1, alpha=0.5, label='Bolthausen-Sznitman')
        plt.title(stat_list[i])
        plt.legend(loc='upper right')
        plt.show()

def plot_SVC_decision_function_histogram(SVC_dec, k_dec, b_dec):
    """
    plots histogram for the decision function produced by SVC
    @param SVC_dec : 1-d Array - refer to return of get_decision_function
    @param k_dec   : 1-d Array - refer to return of get_decision_function
    @param b_dec   : 1-d Array - refer to return of get_decision_function
    """
    plt.figure()
    bins = np.linspace(np.ceil(np.amin(SVC_dec)) - 10, np.ceil(np.amax(SVC_dec)) + 10, 100)
    plt.hist(k_dec, bins, facecolor='magenta', alpha=0.5, label='Kingman')
    plt.hist(b_dec, bins, facecolor='cyan', alpha=0.5, label='Bolthausen-Sznitman')
    plt.title('Frequency of Decision Function Values')
    plt.xlabel('Decision Function Value')
    plt.ylabel('Frequencies')
    plt.legend(loc='upper right')
    plt.show()

def plot_ROC_curve(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    plots ROC Curve
    @param X_train_scaled : 2-d Array  - refer to return of scale_X
    @param X_test_scaled  : 2-d Array  - refer to return of scale_X
    @param y_train        : 1-d Array  - refer to return of preprocess_data
    @param y_test         : 1-d Array  - refer to return of preprocess_data
    """
    lr_clf = LogisticRegression()
    lr_clf.fit(X_train_scaled, y_train)
    pred_ROC = lr_clf.predict_proba(X_test_scaled)
    false_positive_rate, recall, thresholds = metrics.roc_curve(y_test, pred_ROC[:, 1])
    roc_auc = metrics.auc(false_positive_rate, recall)

    plt.figure()
    plt.title('Receiver Operating Chraracteristic (ROC) Curve')
    plt.plot(false_positive_rate, recall, 'b', label='AUC = {:.2f}'.format(roc_auc))
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Fall-Out')
    plt.ylabel('Recall')
    plt.show()

def perform_pca(SVC_dec, X_test_raw, y_test, coef, model_list, three_d=False):
    """
    performs pca to n_comp number of components and plots the 2-d result
    @param X_train_raw : 2-d Array - refer to return of preprocess_data
    @param X_test_raw  : 2-d Array - refer to return of preprocess_data
    @param y           : 1-d Array - refer to return of preprocess_data
    @param coef        : Int       - refer to return of define_classifier
    @param n_comp      : Int       - number of principal components to keep
    """
    pca = decomposition.PCA(n_components=1)  # 7 features
    pca_X = np.zeros_like(X_test_raw)
    for i in range(len(pca_X)):
        pca_X[i] = project_onto_plane(coef, X_test_raw[:][i])
    dec_pca = pca.fit_transform(pca_X).ravel()

    plt.figure()
    for i in range(len(model_list)):
        xs = SVC_dec[y_test == i]
        ys = dec_pca[y_test == i]
        plt.scatter(xs, ys, c=COLORS[i], label=model_list[i])
    plt.title('PCA 2D')
    plt.legend(loc='upper right')
    plt.show()

    # optional
    if three_d:
        pca = decomposition.PCA(n_components=2)  # 7 features
        pca_X = np.zeros_like(X_test_raw)
        for i in range(len(pca_X)):
            pca_X[i] = project_onto_plane(coef, X_test_raw[:][i])
        dec_pca = pca.fit_transform(pca_X)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.gca(projection='3d')
        plt.rcParams['legend.fontsize'] = 10
        for i in range(len(model_list)):
            xs = dec_pca[:, 0][y_test == i]
            ys = dec_pca[:, 1][y_test == i]
            zs = SVC_dec[y_test == i]
            ax.scatter(xs, ys, zs, alpha=0.5, c=COLORS[i], label=model_list[i])
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        ax.set_zlabel('Hyperplane Decision Function')
        plt.title('PCA 3D')
        ax.legend(bbox_to_anchor=(0.35, 0.9))
        plt.show()
