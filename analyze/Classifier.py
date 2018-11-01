#! /usr/bin/python
# -*- coding: utf-8 -*-
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition, metrics, preprocessing
from sklearn.svm import SVC
# from sklearn.cross_validation import train_test_split
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split

from utils import project_onto_plane


__author__ = 'Jayeol Chun'


class Classifier(object):
  """
  receives data of form:
  {
    model1 : [ (data1, data2, ...datan) ],
    model2 : [ (data1, data2, ...datan) ]
  }
  """
  def __init__(self, data, kernel="linear"):
    print("\nInitiating classifier..")
    self.color_list = ['magenta', 'cyan']
    self.clf = SVC(kernel=kernel)
    # self.scaler = preprocessing.StandardScaler()
    # X_trn, X_tst, y_trn, y_tst = self._preprocess(data)
    X_trn, X_tst, y_trn, y_tst = self.preprocess(data)

    coef, intercept = self.fit(X_trn, y_trn)
    self.dec = self.clf.decision_function(X_tst) # distance of samples from separaing hyperplane
    k_dec, b_dec = self.dec[y_tst==0], self.dec[y_tst==1]

    # print ("Coef:", coef)
    # print("Intecp:", intercept)
    # print("\ndec func:")
    # print(self.dec)
    # print("k dec:")
    # print(k_dec)
    # print("b dec:")
    # print(b_dec)

    tst_score = self.test(X_tst, y_tst)

    print("test Score:", tst_score)

    self.plot_desc_func_histogram(k_dec, b_dec)

    self.plot_ROC_curve(X_trn, X_tst, y_trn, y_tst)

    self.pca(y_tst, coef)


  def fit(self, X_trn, y_trn):
    self.clf.fit(X_trn, y_trn)
    coef, intercept = self.clf.coef_[0], self.clf.intercept_[0]
    return coef, intercept

  def test(self, X_tst, y_tst):
    y_pred = self.clf.predict(X_tst)
    # print("Pred:", y_pred)
    return metrics.accuracy_score(y_tst, y_pred)

  def predict(self):
    pass

  def preprocess(self, data, test_size=.20):
    self.index = {}
    _data = []
    for i, (modelName, datum) in enumerate(data.items()):
      self.index[modelName] = i
      _data.append(datum)
    n = len(_data[0])
    # print("N:", n)
    # self.X_raw = np.asarray(_data, dtype=np.float).reshape(n, -1)
    # self.X_raw = np.array(_data, dtype=np.float)
    raw = np.array(_data, dtype=np.float)
    self.X_raw = np.append(raw[0], raw[1], axis=0)
    self.y_raw = np.array([np.zeros(n), np.ones(n)]).ravel()

    # print("--")
    # print(self.X_raw)
    # print(self.y_raw)

    # print("---")
    X_trn, X_tst, y_trn, y_tst = train_test_split(self.X_raw, self.y_raw, test_size=test_size)

    self.X_trn_raw = X_trn
    self.X_tst_raw = X_tst

    self.scalar = preprocessing.StandardScaler().fit(X_trn)
    X_trn_scaled = self.scalar.transform(X_trn)
    X_tst_scaled = self.scalar.transform(X_tst)

    # print("--")
    # print(X_trn)
    # print(X_tst)

    return X_trn_scaled, X_tst_scaled, y_trn, y_tst

  def plot_desc_func_histogram(self, k_dec, b_dec):
    plt.figure()
    bins = np.linspace(np.ceil(np.amin(self.dec))-10, np.ceil(np.amax(self.dec))+10, 100)
    plt.hist(k_dec, bins, facecolor=self.color_list[0], alpha=0.5, label='Kingman')
    plt.hist(b_dec, bins, facecolor=self.color_list[1], alpha=0.5, label='Bolthausen-Sznitman')
    plt.title('Frequency of Decision Function Values')
    plt.xlabel('Decision Function Value')
    plt.ylabel('Frequencies')
    plt.legend(loc='upper right')
    plt.show()

  def plot_ROC_curve(self, X_train_scaled, X_test_scaled, y_train, y_test):
    """
    plots ROC Curve
    @param X_train_scaled : 2-d Array  - refer to return of __scale_X
    @param X_test_scaled  : 2-d Array  - refer to return of __scale_X
    @param y_train        : 1-d Array  - refer to return of __preprocess_data
    @param y_test         : 1-d Array  - refer to return of __preprocess_data
    """
    lr_clf = LogisticRegression()
    lr_clf.fit(X_train_scaled, y_train)
    pred_ROC = lr_clf.predict_proba(X_test_scaled)
    false_positive_rate, recall, thresholds = metrics.roc_curve(y_test, pred_ROC[:, 1])
    roc_auc = metrics.auc(false_positive_rate, recall)

    plt.figure()
    plt.title('ROC Curve')
    plt.plot(false_positive_rate, recall, 'b', label='AUC = {:.2f}'.format(roc_auc))
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Fall-Out')
    plt.ylabel('Recall')
    plt.show()

  def pca(self, y_test, coef, three_d=False):  # edit comments
    """
    performs PCA and plots the 2-d result
    @param SVC_dec    : 2-d Array - refer to return of __get_decision_function
    @param X_test_raw : 2-d Array - refer to return of __preprocess_data
    @param y_test     : 1-d Array - refer to return of __preprocess_data
    @param coef       : Tuple     - refer to return of __define_classifier
    @param three_d    : Bool      - refer to return of perform_ml
    """
    pca = decomposition.PCA(n_components=2)
    pca_X = np.zeros_like(self.X_tst_raw)
    for i in range(len(pca_X)):
      pca_X[i] = project_onto_plane(coef, self.X_tst_raw[:][i])
    dec_pca = pca.fit_transform(pca_X)

    plt.figure()
    # for i in range(len(model_list)):
    #   xs = dec_pca[:, 0][y_test == i]
    #   ys = self.dec[y_test == i]
    #   plt.scatter(xs, ys, c=color_list[i], label=model_list[i])
    k_xs = dec_pca[:, 0][y_test == 0]
    k_ys = self.dec[y_test == 0]
    plt.scatter(k_xs, k_ys, c=self.color_list[0], label='Kingman')

    b_xs = dec_pca[:, 0][y_test == 1]
    b_ys = self.dec[y_test == 1]
    plt.scatter(b_xs, b_ys, c=self.color_list[1], label='Bolthausen-Sznitman')

    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.title('PCA 2D')
    plt.legend(loc='upper right')
    plt.show()

    # optional
    # if three_d:
    #   fig = plt.figure(figsize=(10, 8))
    #   ax = fig.gca(projection='3d')
    #   plt.rcParams['legend.fontsize'] = 10
    #   for i in range(len(model_list)):
    #     xs = dec_pca[:, 0][y_test == i]
    #     ys = dec_pca[:, 1][y_test == i]
    #     zs = self.dec[y_test == i]
    #     ax.scatter(xs, ys, zs, alpha=0.5, c=color_list[i], label=model_list[i])
    #   ax.set_xlabel('First Principal Component')
    #   ax.set_ylabel('Second Principal Component')
    #   ax.set_zlabel('SVM Hyperplane Decision Function')
    #   plt.title('PCA 3D')
    #   ax.legend(bbox_to_anchor=(0.35, 0.9))
    #   plt.show()
    #
