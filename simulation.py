# -*- coding: utf-8 -*-
"""
@author: CheYeol
"""

'''
PCAPCAPCAPCA
1. try only the tip for branch length ratio
    -> bottom vs. top??
    currently top

    lets try both
2. Most common num of mutations for a given tree
3. histograms of each statistics: exponential dist?


'''

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import lstsq
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing, metrics, decomposition
from sklearn.linear_model.logistic import LogisticRegression
import matplotlib.pyplot as plt
from matplotlib import cm
import coal_sims_def as cs_def

# Print Options
np.set_printoptions(threshold=10)
np.set_printoptions(precision=6)

# Parameters
sample_size = 6
n = 500
mu = 1.75
draw = False
stat_show = False
if n <= 4:
    draw = True
    stat_show = False
cs_def.sample_size, cs_def.mu = sample_size, mu

# Data Structure
model_list = ['Kingman', 'Bolthausen-Sznitman']
colors_list = ['magenta', 'cyan']
stat_list = ['Average Number of Mutations', 'Average Number of Ancestors',
              'Mean Separation Time', 'Average Height', 'Top Branch Length',
              'Bottom Branch Length',
              'Measure of Asymmetry using Variance Only',
              'Most frequently Observed Mutation Value']
m = len(stat_list)  # Number of Parameters(Features)
k_list, b_list = np.zeros((n, m)), np.zeros((n, m))  # Data Storage

###############################################################################

# main instructions: Execute n times
for p in range(0, n):
    # Set-Up
    kingman_coalescent_list = np.empty(sample_size, dtype=np.object)
    bs_coalescent_list = np.empty(sample_size, dtype=np.object)

    # Populate the lists with samples
    cs_def.populate_coalescent_list(kingman_coalescent_list, bs_coalescent_list)

    # Kingman Coalescent Model Tree Construction
    kingman_coalescent_list = cs_def.kingman_coalescence(kingman_coalescent_list,
                                                         *(k_list, p))

    # Bolthausen-Sznitman Coalescent Model Tree Construction
    bs_coalescent_list = cs_def.bs_coalescence(bs_coalescent_list, *(b_list, p))

    kingman_ancestor, bs_ancestor = kingman_coalescent_list[0], bs_coalescent_list[0]
    kingman_ancestor.identity, bs_ancestor.identity = 'K', 'B'
    kingman_coalescent_list, bs_coalescent_list = None, None

    # Only if the option to display the tree has been enabled
    if draw:
        cs_def.display_tree(kingman_ancestor)
        cs_def.display_tree(bs_ancestor)

# Data Analysis & Display
data_k, data_b = np.transpose(k_list), np.transpose(b_list)
k_stats, b_stats = np.zeros((2, m)), np.zeros((2, m))
k_stats[0], b_stats[0] = np.mean(data_k, axis=1), np.mean(data_b, axis=1)
k_stats[1], b_stats[1] = np.std(data_k, axis=1), np.std(data_b, axis=1)

print("\n<<Tree Statistics>> with {:d} Trees Each with Standard Deviation".format(n))
for model_name, means, stds in zip(model_list, (k_stats[0], b_stats[0]), (k_stats[1], b_stats[1])):
    for stat_label, mean, std in zip(stat_list, means, stds):
        print(model_name, stat_label, ":", mean, ",", std)
    print()
print("((Kingman vs. Bolthausen-Sznitman)) Side-by-Side Comparison :")
for i in range(0, m):
    print(stat_list[i], ":\n", k_stats[0][i], " vs.", b_stats[0][i],
          "\n", k_stats[1][i], "    ", b_stats[1][i])
print()
model_name, mean, means, stat_label, std, stds = None, None, None, None, None, None

###############################################################################

if stat_show:

    # Training for Prediction
    k_label, b_label = np.zeros(n), np.ones(n)
    X, y = np.append(k_list, b_list, axis=0), np.append(k_label, b_label, axis=0)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.50)
    scaler = preprocessing.StandardScaler().fit(X_train_raw, y_train)
    X_train_scaled = scaler.transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    SVC_clf = svm.SVC(kernel='linear')
    SVC_clf.fit(X_train_scaled, y_train)

    SVC_dec = SVC_clf.decision_function(X_test_scaled)
    k_dec, b_dec = SVC_dec[y_test==0], SVC_dec[y_test==1]
    print(model_list[0], "Decision Function Mean:", np.mean(k_dec))
    print(model_list[1], "Decision Function Mean:", np.mean(b_dec))
    y_train_pred = SVC_clf.predict(X_train_scaled)
    print("Train Set Accuracy:", metrics.accuracy_score(y_train, y_train_pred))
    y_pred = SVC_clf.predict(X_test_scaled)
    print("Test Set Accuracy:", metrics.accuracy_score(y_test, y_pred))

    # Histogram
    plt.figure()
    bins = np.linspace(np.ceil(np.amin(SVC_dec))-10, np.ceil(np.amax(SVC_dec))+10, 100)
    plt.hist(k_dec, bins, facecolor=colors_list[0], alpha=0.5, label='Kingman')
    plt.hist(b_dec, bins, facecolor=colors_list[1], alpha=0.5, label='BSF')
    plt.title('Frequency of Decision Function Values: {:d} Executions'.format(n))
    plt.xlabel('Decision Function Value')
    plt.ylabel('Frequencies')
    plt.legend(loc='upper right')
    plt.show()

    # ROC Curve
    lr_clf = LogisticRegression()
    lr_clf.fit(X_train_scaled, y_train)
    pred_ROC = lr_clf.predict_proba(X_test_scaled)
    false_positive_rate, recall, thresholds = metrics.roc_curve(y_test, pred_ROC[:, 1])
    roc_auc = metrics.auc(false_positive_rate, recall)
    lr_clf, pred_ROC, thresholds = None, None, None

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

    # Analysis of the Decision Function of SVM
    coef, intercept = SVC_clf.coef_[0], SVC_clf.intercept_[0]
    print("Coefficients:", coef)
    print("Intercept:", intercept)

    # Least Squares Solution
    X_scaled = np.append(X_train_scaled, X_test_scaled, axis=0)
    X_scaled_transpose = np.transpose(X_scaled)
    X_scaled_lstsq = lstsq(X_scaled_transpose, coef)[0]
    lstsq_dec = SVC_clf.decision_function(X_scaled)
    lstsq_y = np.append(y_train, y_test, axis=0)

    for i in range(len(model_list)):
        xs = X_scaled_lstsq[lstsq_y==i]
        ys = lstsq_dec[lstsq_y==i]
        plt.scatter(xs, ys, c=colors_list[i], label=model_list[i])
    plt.title('LSTSQ')
    plt.show()

    #X_test_scaled_transpose = np.transpose(X_test_scaled)
    #X_test_scaled_lstsq = lstsq(X_test_scaled_transposem coef)[0]
    #lstsq_dec = SVC_clf.decision_function

    # PCA
    pca = decomposition.PCA(n_components=2)  # 2 classes
    #pca_X = np.vstack((X_scaled_lstsq, lstsq_dec)).T
    dec_pca = pca.fit_transform(X)

    plt.figure()
    for i in range(len(model_list)):
        xs = dec_pca[:, 0][y==i]
        ys = dec_pca[:, 1][y==i]
        plt.scatter(xs, ys, c=colors_list[i], label=model_list[i])
    plt.title('Pure PCA')
    plt.legend(loc='upper right')
    plt.show()

    pca_X = np.vstack((X_scaled_lstsq, lstsq_dec)).T
    dec_pca = pca.fit_transform(pca_X)

    plt.figure()
    for i in range(len(model_list)):
        xs = dec_pca[:, 0][lstsq_y==i]
        ys = dec_pca[:, 1][lstsq_y==i]
        plt.scatter(xs, ys, c=colors_list[i], label=model_list[i])
    plt.title('LSTSQ PCA')
    plt.legend(loc='upper right')
    plt.show()




    '''
    # 2-d PCA
    pca = decomposition.PCA(n_components=2)
    dec_pca = pca.fit_transform(np.append(X_train_raw, X_test_raw,axis=0))

    plt.figure()
    for i in range(len(model_list)):
        xs = dec_pca[:, 0][y == i]
        ys = dec_pca[:, 1][y == i]
        plt.scatter(xs, ys, c=colors_list[i], label=model_list[i])
    plt.xlabel("")
    plt.ylabel("")
    plt.legend(loc='upper right')
    plt.title('PCA')
    plt.show()

    # 3-d PCA
    pca = decomposition.PCA(n_components=3)
    dec_pca = pca.fit_transform(np.append(X_train_raw, X_test_raw,axis=0))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.gca(projection='3d')
    plt.rcParams['legend.fontsize'] = 10
    for i in range(len(model_list)):
        xs = dec_pca[:, 0][y == i]
        ys = dec_pca[:, 1][y == i]
        zs = dec_pca[:, 2][y == i]
        plt.scatter(xs, ys, zs ,alpha=0.5, c=colors_list[i], label=model_list[i])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.legend(bbox_to_anchor=(0.35, 0.9))
    plt.title('PCA')
    plt.show()


    # 3-D Graph
    fig = plt.figure(figsize=(10, 8))
    ax = fig.gca(projection='3d')
    plt.rcParams['legend.fontsize'] = 10
    for i in range(len(model_list)):
        xs = X_test_scaled[:, 0][y_test == i]
        ys = X_test_scaled[:, 1][y_test == i]
        zs = X_test_scaled[:, 2][y_test == i]
        ax.scatter(xs, ys, zs, alpha=0.5, c=colors_list[i], label=model_list[i])
    x_min, x_max = X_test_scaled[:, 0].min() - .5, X_test_scaled[:, 0].max() + .5
    y_min, y_max = X_test_scaled[:, 1].min() - .5, X_test_scaled[:, 1].max() + .5
    num_linspace = 30
    xx, yy = np.linspace(x_min, x_max, num_linspace), np.linspace(y_min, y_max, num_linspace)
    xx, yy = np.meshgrid(xx, yy)
    zz = (- intercept - xx * coef[0] - yy * coef[1]) / coef[2]  # Separating Hyperplane Equation
    ax.plot_surface(xx, yy, zz, rstride=num_linspace, cstride=num_linspace, cmap=cm.hot, alpha=0.4)
    ax.set_xlabel('Number of Mutations')
    ax.set_ylabel('Number of Ancestors')
    ax.set_zlabel('Heterozygosity')
    plt.title('Linear Classification: {:d} Sample Size & {:.2f} Mu'.format(sample_size, mu))
    ax.legend(bbox_to_anchor=(0.35, 0.9))
    plt.show()

    '''

