# -*- coding: utf-8 -*-                     #
# ========================================= #
# Coalescent Simulations APIs               #
# author      : Che Yeol (Jayeol) Chun      #
# last update : 06/26/2016                  #
# ========================================= #

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from Bio import Phylo
from io import StringIO
from scipy.stats import poisson
from sklearn import preprocessing, metrics, decomposition
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.linear_model.logistic import LogisticRegression
import matplotlib.pyplot as plt
import sys

########################### Global Variables ###########################

# will be initialized as defined in main program
model_list     = None
color_list     = None
stat_list      = None
param_list     = None
sample_size    = 0
n              = 0
mu             = 0
m              = 0
num_desc_thold = 0

########################### Tree Nodes ###########################

class Sample:  # Leaf of Tree
    def __init__(self, identity_count):
        """
        @param identity_count: Int - unique ID number to distinguish this sample from the rest
        """
        self.identity = identity_count   # unique identity of each sample
        self.big_pivot = identity_count  # Conforms to usual visualization
        self.next = None                 # links to its left neighbor child of its parent
        self.time = 0                    # time to the previous coalescent event
        self.generation = 0              # each coalescent event represents a generation, beginning from the bottom of the tree
        self.mutations = 0               # mutations that occurred from the previous coalescent event

    def __repr__(self):
        return 'Sample {} with Mutations {:d}.'.format(self.identity, self.mutations)

    def is_sample(self):
        return True

class Ancestors(Sample):  # Internal Node of Tree, inherits Sample
    def __init__(self, identity_count):
        """
        @param identity_count: Int - unique ID number to distinguish this ancestor from the rest
        """
        super(self.__class__, self).__init__(identity_count)
        self.identity = 'A{}'.format(identity_count)
        self.generation = identity_count
        self.height = 0
        self.left = None                    # left-most child
        self.right = None                   # right-most child
        self.descendent_list = np.zeros(0)  # all samples below it
        self.children_list = np.zeros(0)    # all children directly below it

    def __repr__(self):
        return 'Ancestor {} with Mutations {:d}.'.format(self.identity, self.mutations)

    def is_sample(self):
        return False

########################### External Methods: Main Simulation ###########################

def kingman_coalescence(coalescent_list, *data):
    """
    models the Kingman coalescence
    @param coalescent_list     : 1-d Array - initial coalescent list
    @param data                : Tuple     - (data_list, data_index) -> refer to __update_data
    @return coalescent_list[0] : Ancestor  - root of the Kingman tree
    """
    gen_time = np.zeros(sample_size-1)  # coalescent time for each generation
    nth_coalescence = 0  # Starting from 0

    # Until reaching the Most Recent Common Ancestor
    while np.size(coalescent_list) > 1:
        # Time Calculation
        time = np.random.exponential(1/__k_F(np.size(coalescent_list)))
        gen_time[nth_coalescence] = time
        nth_coalescence += 1

        # merged ancestor of the coalescent event and its children obtained
        consts = {'identity_count' : nth_coalescence}
        coalescent_list, ancestor, children_list = __coalesce_children(coalescent_list, **consts )
        # update the tree using mutations as branch length, recording data along the way
        lists = {'coalescent_list' : coalescent_list, 'children_list' : children_list, 'gen_time' : gen_time}
        coalescent_list = __update_children(ancestor, *data, **lists)
    coalescent_list[0].identity = 'K'
    #print(coalescent_list[0], "This tree's branch length under threshold is:", data[0][data[1]][3])
    return coalescent_list[0]

def bs_coalescence(coalescent_list, *data):
    """
    models the Bolthausen-Sznitman coalescence
    @param coalescent_list     : 1-d Array - initial coalescent list
    @param data                : Tuple     - (data_list, data_index) -> refer to __update_data
    @return coalescent_list[0] : Ancestor  - root of the Bolthausen-Sznitman tree
    """
    gen_time = np.zeros(sample_size-1)  # coalescent time for each generation
    nth_coalescence = 0  # Starting from 0

    # Until reaching the Most Recent Common Ancestor
    while np.size(coalescent_list) > 1:
        # Time and Number of Children Calculation
        m_list = np.arange(2, np.size(coalescent_list)+1)
        mn_rate = np.zeros(np.size(coalescent_list)-1)
        bsf_rate = np.zeros(np.size(coalescent_list)-1)
        mn_rate, total_rate = __b_F(mn_rate, np.size(coalescent_list))
        for j in range(0, np.size(mn_rate)):
            bsf_rate[j] = mn_rate[j] / total_rate
        num_children = np.random.choice(m_list, 1, replace=False, p=bsf_rate)
        time = np.random.exponential(1/total_rate)
        gen_time[nth_coalescence] = time
        nth_coalescence += 1

        # merged ancestor of the coalescent event and its children obtained
        consts = {'identity_count' : nth_coalescence, 'num_children' : num_children}
        coalescent_list, ancestor, children_list = __coalesce_children(coalescent_list, **consts)
        # update the tree using mutations as branch length, recording data along the way
        lists = {'coalescent_list' : coalescent_list, 'children_list' : children_list, 'gen_time' : gen_time}
        coalescent_list = __update_children(ancestor, *data, **lists)
    coalescent_list[0].identity = 'B'
    #print(coalescent_list[0], "This tree's branch length under threshold is:", data[0][data[1]][3])
    return coalescent_list[0]

########################### Internal Methods: Main Simulation ###########################

def __k_F(n):
    """
    computes the Kingman function
    @param n : Int   - current size of the coalescent list, i.e. number of samples available for coalescence
    @return  : Float - Kingman function result
    """
    return n*(n-1) / 2

def __b_F(mn_rate, n):  # more detailed description needed
    """
    computes the Bolthausen-Sznitman function
    @param mn_rate : 1-d Array -
    @param n       : Int       - current size of the coalescent list, i.e. number of samples available for coalescence
    @return        : Tuple     - ( mn_rate    : 1-d Array -
                                   total_rate : 1-d Array - holds each i rate )
    """
    total_rate = 0
    for i in range(n-1):
        m = i + 2
        i_rate = n / (m * (m-1))
        mn_rate[i] = i_rate
        total_rate += i_rate
    return mn_rate, total_rate

def __update_children(ancestor, data_list, data_index, coalescent_list, children_list, gen_time):
    """
    A.for each child node under the ancestor, do:
        1) calculate its time, taking into account the generation difference between the sample and its ancestor
        2) based on 1), calculate its mutation
        3) perform appropriate tasks depending on what type the child is -> refer to comments below for details
    B. update the ancestor
    @param ancestor         : Ancestor  - newly merged ancestor
    @param data_list        : 2-d Array - to be updated
    @param data_index       : Int       - ensures each data is stored at right place
    @param coalescent_list  : 1-d Array - initial coalescent list
    @param children_list    : 1-d Array - for analysis of each child in the list
    @param gen_time         : 1-d Array - used to update time of each child to the ancestor
    @return coalescent_list : 1-d Array - updated coalescent list
    """
    temp_list = np.copy(children_list)

    # children_index: index of changing children_list
    children_index = 0

    ##########################################################################################
    ### BEGIN: iteration through the children_list
    while children_index < np.size(temp_list):
        # current child under inspection
        current = temp_list[children_index]

        __update_time(current, ancestor, gen_time)
        current.mutations = poisson.rvs(mu * current.time)

        # First Case : a Sample(Leaf)
        if current.is_sample():
            __update_data(data_list, data_index, *zip((0, 2), (current.mutations, current.mutations)))

        # Second Case : an Internal Node with Mutations == 0
        elif not (current.is_sample()) and current.mutations == 0:
            # Delete this Current Child from the Coalescent List
            cond = coalescent_list == temp_list[children_index]
            coalescent_list = np.delete(coalescent_list, int(np.where(cond)[0]))

            # Replace this Current Child with its children nodes
            temp_list = np.insert(temp_list, children_index, current.children_list)
            temp_list = np.delete(temp_list, children_index + np.size(current.children_list))

            # Create Linked List that Connects the Replacing children_list with the original children on its left if it exists
            if children_index > 0:
                temp_list[children_index].next = temp_list[children_index-1]
            # Increase the index appropriately by jumping over the current child's children
            children_index += (np.size(current.children_list) - 1)

        # Third Case : an Internal Node with Mutations > 0
        else:
            if len(current.descendent_list) >= np.around(sample_size * num_desc_thold):
                __update_data(data_list, data_index, *zip((0, 1, 3), (current.mutations, 1, current.mutations)))
            else:
                __update_data(data_list, data_index, *zip((0, 1), (current.mutations, 1)))

        # Delete Current Child from the Coalescent List (unless Deleted alrdy in the Second Case)
        cond = coalescent_list == temp_list[children_index]
        if True in cond:        coalescent_list = np.delete(coalescent_list, int(np.where(cond)[0]))

        # Link current child to its left
        if children_index > 0:  current.next = temp_list[children_index - 1]

        # increase indices
        children_index += 1
    ### END: iteration through the children_list
    ##########################################################################################

    # Update new information to the ancestor
    __update_ancestor(ancestor, temp_list)
    return coalescent_list

def __coalesce_children(coalescent_list, identity_count, num_children=2):
    """
    Given a number of children to be merged, perform a coalescent event that creates a merged ancestor
    @param coalescent_list : 1-d Array - holds samples currently available for new coalescence
    @param identity_count  : Int       - distinct ID number to create a new merged sample
    @param num_children    : Int       - number of children to be coalesced
    @return                : Tuple     - ( coalescent_list : 1-d Array - updated coalescent list
                                           merge_sample    : Ancestor  - new merged sample
                                           children_list   : 1-d Array - Ancestor's direct children )
    """
    # Create an Internal Node Representing a Coalescent Event
    merge_sample = Ancestors(identity_count)

    # The merge_sample's immediate children chosen
    children_list = np.random.choice(coalescent_list, num_children, replace=False)
    __quicksort(children_list, 0, np.size(children_list)-1)  # sorted for visual ease
    coalescent_list = np.append(coalescent_list, merge_sample)
    return coalescent_list, merge_sample, children_list

def __update_descendent_list(children_list):
    """
    creates a descendent list by replacing samples in the children list with its own descendent list
    @param children_list    : 1-d Array - for each children in the list, see what samples are below it and compile them
    @return descendent_list : 1-d Array - newly created descendent_list
    """
    descendent_list = np.copy(children_list)
    i = 0
    while i < np.size(descendent_list):
        if not(descendent_list[i].is_sample()):
            # insert the internal node's own descendent list at the node's index in the current descendent_list
            # -> since the node below the sample, its descdent list has already been updated
            size = np.size(descendent_list[i].descendent_list)
            descendent_list = np.insert(descendent_list, i, descendent_list[i].descendent_list)
            # remove the given internal node from the descendent list -> we only want the samples, not the internal nodes
            descendent_list = np.delete(descendent_list, i+size)
            i += size
        else:  # if sample
            i += 1  # move to the next on the descendent list
    return descendent_list

def __update_time(sample, ancestor, gen_time):
    """
    adds up the between-generation time
    @param sample   : Ancestor / Sample - sample whose coalescent time to its ancestor is to be calculated
    @param ancestor : Ancestor          - newly merged ancestor
    @param gen_time : 1-d Array         - holds coalescent time between generations
    """
    for j in range(ancestor.generation - 1, sample.generation - 1, -1):
        sample.time += gen_time[j]

def __update_ancestor(ancestor, children_list):
    """
    assigns new attributes to the merged ancestor
    @param ancestor      : Ancestor  - newly merged ancestor, represents a single coalescent event
    @param children_list : 1-d Array - nodes that are derived from the ancestor
    """
    ancestor.children_list = children_list
    ancestor.descendent_list = __update_descendent_list(children_list)
    ancestor.right = children_list[np.size(children_list) - 1]
    ancestor.big_pivot = ancestor.right.big_pivot
    ancestor.left = ancestor.children_list[0]

def __update_data(data_list, data_index, *data):
    """
    updates the data list
    @param data_list  : 2-d Array - holds overall data
    @param data_index : Int       - ensures each data is stored at right place
    @param data       : Tuple     - (index, value) where the value is to be added to the data_list at the index
    """
    for index, value in data:
        data_list[data_index][index] += value

########################### External Methods: Statistics and Plots ###########################

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

def plot_SVC_decision_function_histogram(SVC_dec, k_dec, b_dec):
    """
    plots histogram for the decision function produced by SVC
    @param SVC_dec : 1-d Array - refer to return of get_decision_function
    @param k_dec   : 1-d Array - refer to return of get_decision_function
    @param b_dec   : 1-d Array - refer to return of get_decision_function
    """
    plt.figure()
    bins = np.linspace(np.ceil(np.amin(SVC_dec)) - 10, np.ceil(np.amax(SVC_dec)) + 10, 100)
    plt.hist(k_dec, bins, facecolor=color_list[0], alpha=0.5, label='Kingman')
    plt.hist(b_dec, bins, facecolor=color_list[1], alpha=0.5, label='Bolthausen-Sznitman')
    plt.title('Frequency of Decision Function Values: {:d} Executions'.format(n))
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

def perform_pca(SVC_dec, X_test_raw, y_test, coef, three_d=False):  # edit comments
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
        pca_X[i] = __project_onto_plane(coef, X_test_raw[:][i])
    dec_pca = pca.fit_transform(pca_X).ravel()

    plt.figure()
    for i in range(len(model_list)):
        xs = SVC_dec[y_test == i]
        ys = dec_pca[y_test == i]
        plt.scatter(xs, ys, c=color_list[i], label=model_list[i])
    plt.title('PCA 2D')
    plt.legend(loc='upper right')
    plt.show()

    # optional
    if three_d:
        pca = decomposition.PCA(n_components=2)  # 7 features
        pca_X = np.zeros_like(X_test_raw)
        for i in range(len(pca_X)):
            pca_X[i] = __project_onto_plane(coef, X_test_raw[:][i])
        dec_pca = pca.fit_transform(pca_X)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.gca(projection='3d')
        plt.rcParams['legend.fontsize'] = 10
        for i in range(len(model_list)):
            xs = dec_pca[:, 0][y_test == i]
            ys = dec_pca[:, 1][y_test == i]
            zs = SVC_dec[y_test == i]
            ax.scatter(xs, ys, zs, alpha=0.5, c=color_list[i], label=model_list[i])
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        ax.set_zlabel('Hyperplane Decision Function')
        plt.title('PCA 3D')
        ax.legend(bbox_to_anchor=(0.35, 0.9))
        plt.show()

########################### External Methods: Miscellaneous ###########################

def set_parameters(default_val):
    """
    sets the custom values of parameters by receiving user inputs, if instructed to do so
    @param default_val : Tuple - default initial parameter values
    @return            : List  - returns (possibly changed) parameter values
    """
    direction = str(input("Run with default values? \"custom\" for custom parameters values : "))
    return_val = list(default_val)
    if "custom" in direction or "no" in direction:
        for i, vals in enumerate(zip(param_list, default_val)):
            exec("return_val[i] = __request_user_input(vals)")
    if n > 1000: __check_n()  # warning : n might have been set unnecessarily big for a test-case
    return return_val

def display_init_params():
    print("\n****** Running with: ******")
    for label, val in zip(param_list, (sample_size, n, mu, num_desc_thold)):
        print("   ",label, ":", val)
    print()

def populate_coalescent_list(kingman_coalescent_list, bs_coalescent_list):
    """
    fills each array with samples of both coalescent types
    @param kingman_coalescent_list : 1-d Array - to be filled with samples
    @param bs_coalescent_list      : 1-d Array - to be filled with samples
    """
    for i in range(sample_size):
        kingman_coalescent_list[i] = Sample(i + 1)
        bs_coalescent_list[i] = Sample(i + 1)

def display_stats(data_k, data_b, model_list, stat_list):
    """
    displays the cumulative statistics of all trees observed for Kingman and Bolthausen-Sznitman
    @param data_k     : 2-d Array - holds data extracted from Kingman trees
    @param data_b     : 2-d Array - holds data extracted from Bolthausen-Sznitman trees
    @param model_list : 1-d Array - provides the names of coalescent models
    @param stat_list  : 1-d Array - provides description of each statistics examined
    """
    k_stats, b_stats = np.zeros((2, m)), np.zeros((2, m))
    k_stats[0], b_stats[0] = np.mean(data_k, axis=1), np.mean(data_b, axis=1)
    k_stats[1], b_stats[1] = np.std(data_k, axis=1), np.std(data_b, axis=1)
    print("\n<<Tree Statistics>> with {:d} Trees Each with Standard Deviation".format(n))
    for model_name, means, stds in zip(model_list, (k_stats[0], b_stats[0]), (k_stats[1], b_stats[1])):
        for stat_label, mean, std in zip(stat_list, means, stds):
            print(model_name, stat_label, ":", mean, ",", std)
        print()
    print("<<Kingman vs. Bolthausen-Sznitman>> Side-by-Side Comparison :")
    for i in range(m):
        print(stat_list[i], ":\n", k_stats[0][i], " vs.", b_stats[0][i],
              "\n", k_stats[1][i], "    ", b_stats[1][i])
    print()

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
        plt.hist(data_k[i], facecolor=color_list[0], bins=bins, lw=1, alpha=0.5, label='Kingman')
        plt.hist(data_b[i], facecolor=color_list[1], bins=bins, lw=1, alpha=0.5, label='Bolthausen-Sznitman')
        plt.title(stat_list[i])
        plt.legend(loc='upper right')
        plt.show()

########################### Internal Methods: Miscellaneous ###########################

def __request_user_input(vals, *args, multiple_choice=False):
    """
    handles user input
    @param vals            : List        - holds data name and default data value
    @param args            : Tuple       - pre-defined options for multiple choice
    @param multiple_choice : Bool        - lists options if True
    @return value          : Int / Float - result that matches user input
    """
    accepted_range = " >= 0"
    value_name, data_type = vals[0], str(type(vals[1])).split("\'")[1]
    if multiple_choice:
        print("Available options:")
        _range = ()
        diction = dict(args)
        for arg1, arg2 in args:
            print(arg1, ":", arg2)
            _range += (arg1,)
        accepted_range = " in _range"
    while True:
        value = input("Please write custom value for " + value_name + " (only " + data_type + " accepted): ")
        try:
            value = float(value) if "float" in data_type else int(value)
            if eval(str(value) + accepted_range):
                if multiple_choice:  value = diction[value]
                break
            else:
                print("Value outside of the Accepted Range")
        except ValueError:
            print("Wrong Input, please write", data_type,"for", value_name)
    print("Your input for", value_name, ":", value)
    return value

def __check_n():
    """
    confirms whether the user really desires the current n value
    """
    ans = str(input("Are you sure? Type \"yes\" if so: "))
    if ans == "yes":
        pass
    elif ans == "no":
        print("Exiting..")
        sys.exit(0)
    else:
        counter = 0
        while True:
            ans = str(input("Are you sure? Type \"yes\" if so: "))
            if ans == "yes":
                break
            elif ans == "no":
                print("Exiting..")
                sys.exit(0)
            counter += 1
            if counter >= 1:
                print("Exiting..")
                sys.exit(0)

def __quicksort(children_list, first, last):
    """
    sorts the children_list based on the value of big_pivot
    @param children_list : 1-d Array - target to be sorted
    @param first         : Int       - index of first element
    @param               : Int       - index of last element
    """
    if first < last:
        splitpoint = __partition(children_list, first, last)
        __quicksort(children_list, first, splitpoint - 1)
        __quicksort(children_list, splitpoint + 1, last)

def __partition(children_list, first, last):
    """
    partitions in place
    @param children_list : 1-d Array - target to be sorted
    @param first         : Int       - index of first element
    @param last          : Int       - index of last element
    @return hi           : Int       - index at which partition will occur
    """
    lo, hi = first + 1, last
    piv = children_list[first].big_pivot
    while True:
        while lo <= hi and children_list[lo].big_pivot <= piv:    lo += 1
        while hi >= lo and children_list[hi].big_pivot >= piv:    hi -= 1
        if hi < lo:     break
        else:
            temp = children_list[lo]
            children_list[lo], children_list[hi] = children_list[hi], temp
    if hi == first:     return hi
    part = children_list[first]
    children_list[first], children_list[hi] = children_list[hi], part
    return hi

def __project_onto_plane(a, b):
    """
    finds the vector projection of points onto the hyperplane
    @param a : Tuple - coefficients of the hyperplane
    @param b : Tuple - original vector
    @return  : Tuple - new vector projected onto the hyperplane
    """
    dot = np.dot(a, b) / np.linalg.norm(a)
    p = dot * a / np.linalg.norm(a)
    return b - p

########################### Optional : Display Tool ###########################

def display_tree(ancestors):
    """
    displays the Newick Format in string Newick format and its Phylo visualization
    @param ancestors : 1-d Array - root of the tree to be displayed
    """
    for i in range(len(ancestors)):
        newick = __traversal(ancestors[i])
        tree = Phylo.read(StringIO(str(newick)), 'newick')
        Phylo.draw(tree)
        print(newick)

def __traversal(sample):
    """
    iterates through the tree rooted at the sample recursively in pre-order, building up a Newick format
    @param sample  : Ancestor - root of the tree to be displayed
    @return output : String   - complete newick format
    """
    output = ''
    current = sample.right
    output = __recur_traversal((output + '('), current)
    while current.next != sample.left:
        current = current.next
        output = __recur_traversal(output + ', ', current)
    current = sample.left
    output = __recur_traversal(output + ', ', current) + ')' + str(sample.identity)
    return output

def __recur_traversal(output, sample):
    """
    appends the sample's information to the current Newick format, recursively travelling to the sample's leaves as necessary
    @param output  : String            - incoming newick format to be appended new information
    @param sample  : Ancestor / Sample - provides new information
    @return output : String            - modified newick format
    """
    if sample.is_sample():
        output = output + str(sample.identity) + ':' + str(sample.mutations)
        return output
    current = sample.right
    output = __recur_traversal((output + '('), current)
    while current.next != sample.left:
        current = current.next
        output = __recur_traversal(output + ', ', current)
    current = sample.left
    output = __recur_traversal((output + ', '), current)
    output = output + ')' + str(sample.identity) + ':' + str(sample.mutations)
    return output
