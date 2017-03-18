# -*- coding: utf-8 -*-                     #
# ========================================= #
# Coalescent Simulations Main Helper        #
# author      : Che Yeol (Jayeol) Chun      #
# last update : 03/17/2017                  #
# ========================================= #

import sys
import numpy as np
from scipy.stats import poisson


def select_mode():
    '''
    # when -M or --m specified at command line
    allows user to select a single model
    :return:
    '''
    print("\n*** Option to Specify Coalescent Model:")
    usr = num_trials = 0
    while usr == 0:
        usr = input("Select 1 for Kingman, 2 for Bolthausen-Sznitman:\n")
        try:
            usr = int(usr)
            if usr == 1:
                mode = "K"
                print("\n*** Kingman Coalescent Selected")
                break
            elif usr == 2:
                mode = "B"
                print("\n*** Bolthausen-Sznitman Coalescent Selected")
                break
            usr = 0
        except ValueError:
            pass
        num_trials += 1
        if num_trials >= 5:
            print("\n*** Fatal: Exceeded Max number of attempts\n*** System Exiting...")
            sys.exit(1)
    return mode

def display_init_params():
    print("\n****** Running with: ******")
    for label, val in zip(param_list, (sample_size, n, mu, num_desc_thold)):
        print("   ",label, ":", val)
    print()


########################### Main Simulation ###########################

def iterate(num_iter_each, sample_size_range, init_mu,
            param_list, model_list, color_list, stat_list):
    """
    iterate through every combination of sample size and mutation rates given
    """

    accurate_threshold_bbl, refined_mu = np.zeros_like(sample_size_range, dtype=float), np.zeros_like(sample_size_range, dtype=float)
    pass_percent = .9
    for idx, sample_size in enumerate(sample_size_range):
        mu = init_mu if init_mu < 15 else 200
        while True:
            __display_params(param_list, sample_size, mu)
            threshold_bbl, percent_correct = simulation(sample_size, num_iter_each, mu,
                                                        model_list, color_list, stat_list)
            accurate_threshold_bbl[idx] = threshold_bbl
            refined_mu[idx] = mu
            if percent_correct >= pass_percent:
                print("\n*******Either equal to or over {:.2f}, moving along\n".format(pass_percent))
                break

            mu = mu * (1 + pass_percent - percent_correct)
            if mu >= 10000:
                break

    return accurate_threshold_bbl, refined_mu

def simulation(sample_size, niter, mu,
               model_list, color_list, stat_list,
               newick=False, plot_data=False):
    """
    simulates the Kingman and Bolthausen-Sznitman coalescence
    @return : refer to return of get_threshold
    """
    num_feature = len(stat_list)
    k_list, b_list = np.zeros((niter, num_feature)), np.zeros((niter, num_feature))
    for i in range(niter):
        # Populate the lists with samples
        kingman_coalescent_list, bs_coalescent_list = __init_coalescent_list(sample_size)

        # Kingman Coalescence
        __kingman_coalescence(sample_size, mu,
                              kingman_coalescent_list,
                              *(k_list, i), newick=newick)

        # Bolthausen-Sznitman Coalescence
        __bs_coalescence(sample_size, mu,
                         bs_coalescent_list,
                         *(b_list, i), newick=newick)

    return get_threshold(k_list, b_list)

def get_threshold(k_list, b_list):
    """
    get BBL threshold
    @param k_list: Kingman data
    @param b_list: Bolthausen-Sznitman data
    @return      : threhold values, correct prediction percentage
    """
    total_list = np.concatenate([k_list, b_list])
    total_size = len(total_list)
    k_label, b_label = np.zeros_like(k_list), np.ones_like(b_list)
    total_label = np.concatenate([k_label, b_label])
    result = np.concatenate([total_list, total_label], axis=1)
    sorted_result = result[np.argsort(result[:, 0])]

    threshold_bbl, threshold_idx = 0, 0
    goodness, max_goodness = 0, 0
    goodness_log = []
    for index, res in enumerate(sorted_result):
        i = res[1]
        if i == 0:
            goodness += 1
        else:
            goodness -= 1
        goodness_log.append(goodness)
        if goodness > max_goodness:
            max_goodness = goodness
            threshold_idx = index
            try:
                threshold_bbl = (res[0] + sorted_result[index+1, 0]) / 2
            except IndexError:
                threshold_bbl = res[0]

    pred_label = np.concatenate([np.zeros(threshold_idx+1), np.ones(total_size - (threshold_idx+1))])
    sum_correct = np.sum(pred_label == sorted_result[:, 1])
    percent_correct = sum_correct / total_size
    return threshold_bbl, percent_correct


def all_ancestors(items):
    """
    checks whether all items are of Ancestor type
    """
    return all(isinstance(x, Ancestors) for x in items)

def __init_coalescent_list(sample_size):
    """
    initializes coalescent lists for Kingman and Bolthausen-Sznitman src
    @return: Tuple - ( 1-d Array - Kingman coalescent_list
                       1-d Array - Bolthausen-Sznitman coalescent_list
    """
    return np.array([Sample(i+1) for i in range(sample_size)]), np.array([Sample(i+1) for i in range(sample_size)])





def __update_children(mu, ancestor, data_list, data_index, coalescent_list, children_list, gen_time):
    """
    A.for each child node under the ancestor, do:
        1) calculate its time, taking into account the generation difference between the sample and its ancestor
        2) based on 1), calculate the branch's mutation value
        3) perform appropriate tasks depending on what type the child is -> refer to comments below for details
    B. update the ancestor
    @param mu               : Float     - refer to argument of src
    @param ancestor         : Ancestor  - newly merged ancestor
    @param data_list        : 2-d Array - to be updated
    @param data_index       : Int       - ensures each data is stored at right place
    @param coalescent_list  : 1-d Array - initial coalescent list
    @param children_list    : 1-d Array - for analysis of each child in the list
    @param gen_time         : 1-d Array - used to update time of each child to the ancestor
    @return coalescent_list : 1-d Array - updated coalescent list
    """
    temp_list = np.copy(children_list)
    children_index = 0  # index of changing children_list

    ##########################################################################################
    # BEGIN: iteration through the children_list
    while children_index < np.size(temp_list):
        current = temp_list[children_index]  # current child under inspection

        __update_time(current, ancestor, gen_time)
        current.mutations = poisson.rvs(mu * current.time)

        # First Case : a Sample (Leaf Node)
        if current.is_sample():
            __update_data(data_list, data_index, *zip((0,), (current.mutations,)))

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
            children_index += (np.size(current.children_list)-1)

        # Third Case : an Internal Node with Mutations > 0
        #else:
        #    __update_data(data_list, data_index, *zip((0,), (current.mutations,)))

        # Delete Current Child from the Coalescent List (unless Deleted alrdy in the Second Case)
        cond = coalescent_list == temp_list[children_index]
        if True in cond:
            coalescent_list = np.delete(coalescent_list, int(np.where(cond)[0]))

        # Link current child to its left
        if children_index > 0:
            current.next = temp_list[children_index - 1]

        # increase indices
        children_index += 1
    # END: iteration through the children_list
    ##########################################################################################

    # Update new information to the ancestor
    __update_ancestor(ancestor, temp_list)

    return coalescent_list

def __coalesce_children(coalescent_list, identity_count, num_children=2):
    """
    Given a number of children to be merged, perform a coalescent event that creates a merged ancestor
    @param coalescent_list : 1-d Array - holds samples currently available for new coalescence
    @param identity_count  : Int       - distinct ID number to create a new merged sample
    @param num_children    : Int       - number of children to be coalesced, 2 by default
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
            # -> since the node is below the sample, its descdent list must have already been updated
            size = np.size(descendent_list[i].descendent_list)
            descendent_list = np.insert(descendent_list, i, descendent_list[i].descendent_list)

            # remove the given internal node from the descendent list -> we only want the samples, not the internal nodes
            descendent_list = np.delete(descendent_list, i+size)
            i += size
        else:       # if sample,
            i += 1  # move to the next on the descendent list
    return descendent_list

def __update_time(sample, ancestor, gen_time):
    """
    adds up the between-generation time
    @param sample   : Ancestor / Sample - sample whose coalescent time to its ancestor is to be calculated
    @param ancestor : Ancestor          - newly merged ancestor
    @param gen_time : 1-d Array         - holds coalescent time between generations
    """
    for j in range(ancestor.generation-1, sample.generation-1, -1):
        sample.time += gen_time[j]

def __update_ancestor(ancestor, children_list):
    """
    assigns new attributes to the merged ancestor
    @param ancestor      : Ancestor  - newly merged ancestor, represents a single coalescent event
    @param children_list : 1-d Array - nodes that are derived from the ancestor
    """
    ancestor.children_list = children_list
    ancestor.descendent_list = __update_descendent_list(children_list)
    ancestor.right = children_list[np.size(children_list)-1]
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

def __display_params(param_list, sample_size, mu):
    """
    displays values of initial parameters
    @param param_list  : 1-d Array - list of parameters
    @param sample_size : Int       - refer to argument of src
    @param mu          : Float     - refer to argument of src
    """
    print("\n****** Running with: ******")
    for label, val in zip(param_list, (sample_size, mu)):
        print("   ",label, ":", val)
    print()


# Quick Sort Children

def __quicksort(children_list, first, last):
    """
    sorts the children_list based on the value of big_pivot
    @param children_list : 1-d Array - target to be sorted
    @param first         : Int       - index of first element
    @param last          : Int       - index of last element
    """
    if first < last:
        splitpoint = __partition(children_list, first, last)
        __quicksort(children_list, first, splitpoint-1)
        __quicksort(children_list, splitpoint+1, last)

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
        while lo <= hi and children_list[lo].big_pivot <= piv: lo += 1
        while hi >= lo and children_list[hi].big_pivot >= piv: hi -= 1
        if hi < lo: break
        else:
            temp = children_list[lo]
            children_list[lo], children_list[hi] = children_list[hi], temp
    if hi == first: return hi
    part = children_list[first]
    children_list[first], children_list[hi] = children_list[hi], part
    return hi

