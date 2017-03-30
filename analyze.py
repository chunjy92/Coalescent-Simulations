# -*- coding: utf-8 -*-                     #
# ========================================= #
# Statistics and ML toolkits                #
# author      : Che Yeol (Jayeol) Chun      #
# last update : 03/29/2017                  #
# ========================================= #

import numpy as np
from sklearn import preprocessing, metrics, decomposition
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
# from sklearn.linear_model.logistic import LogisticRegression

__author__ = 'Jayeol Chun'

def preprocess_data(k_list, b_list, test_size):
    """
    collects data into a form usable through scikit-learn stat analysis tools
    @param k_list    : 2-d Array - holds raw Kingman data
    @param b_list    : 2-d Array - holds raw Bolthausen-Sznitman data
    @param test_size : Int       - defines how the data is to be randomly split
    @return          : Tuple     - (raw data collection X, raw prediction label collection y, X to be trained,
                                    X to be tested, label for train data, label for test data)
    """
    k_label, b_label = np.zeros(n), np.ones(n)  # predicted variables, where 0: Kingman, 1 : Bolthausen-Sznitman
    X, y = np.append(k_list, b_list, axis=0), np.append(k_label, b_label, axis=0) # raw collection of data and labels that match
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X, y, X_train_raw, X_test_raw, y_train, y_test

def scale_X(X_train_raw, X_test_raw, y_train):
    """
    define a scaler and scale each X
    @param X_train_raw : 2-d Array - refer to return of preprocess_data
    @param X_test_raw  : 2-d Array - refer to return of preprocess_data
    @param y_train     : 1-d Array - refer to return of preprocess_data
    @return            : Tuple     - ( X_train_scaled : 2-d Array - scaled X train data
                                       X_test_scaled  : 2-d Array - scaled X test data )
    """
    scaler = preprocessing.StandardScaler().fit(X_train_raw, y_train)
    X_train_scaled = scaler.transform(X_train_raw)
    X_test_scaled  = scaler.transform(X_test_raw)
    return X_train_scaled, X_test_scaled

def define_classifier(X_train_scaled, y_train, kernel='linear'):
    """
    defines a classifier, fits to train data and get its properties
    @param X_train_scaled : 2-d Array - refer to return of preprocess_data
    @param y_train        : 1-d Array - refer to return of preprocess_data
    @param kernel         : String    - defines the kernel type of the classifier
    @return               : Tuple     - ( SVC_clf   : Classifier - support vector classifier
                                          coef      : Tuple      - linear coefficients of the separating hyperplane
                                          intercept : Int        - intercept of the hyperplane )
    """
    SVC_clf = SVC(kernel=kernel)
    SVC_clf.fit(X_train_scaled, y_train)
    coef, intercept = SVC_clf.coef_[0], SVC_clf.intercept_[0]
    print("Coefficients:", coef)
    print("Intercept:", intercept)
    return SVC_clf, coef, intercept

def get_decision_function(clf, X_test_scaled, y_test):
    """
    returns the decision functions
    @param clf           : Classifier - refer to return of define_classifier
    @param X_test_scaled : 2-d Array  - refer to return of scale_X
    @param y_test        : 1-d Array  - refer to return of preprocess_data
    @return              : Tuple      - ( SVC_dec : 1-d Array - entire decision function
                                          k_dec   : 1-d Array - Kingman decision function
                                          b_dec   : 1-d Array - Bolthausen-Sznitman decision function )
    """
    SVC_dec      = clf.decision_function(X_test_scaled)
    k_dec, b_dec = SVC_dec[y_test == 0], SVC_dec[y_test == 1]
    print(model_list[0], "Decision Function Mean:", np.mean(k_dec))
    print(model_list[1], "Decision Function Mean:", np.mean(b_dec))
    return SVC_dec, k_dec, b_dec

def test_accuracy(clf, X_train_scaled, y_train, X_test_scaled, y_test):
    """
    tests the accuracy of the classifier, using the test data
    @param clf            : Classifier - refer to return of define_classifier
    @param X_train_scaled : 2-d Array  - refer to return of scale_X
    @param y_train        : 1-d Array  - refer to return of preprocess_data
    @param X_test_scaled  : 2-d Array  - refer to return of scale_X
    @param y_test         : 1-d Array  - refer to return of preprocess_data
    """
    y_train_pred = clf.predict(X_train_scaled)
    print("Train Set Accuracy :", metrics.accuracy_score(y_train, y_train_pred))
    y_pred = clf.predict(X_test_scaled)
    print("Test Set Accuracy  :", metrics.accuracy_score(y_test, y_pred))
