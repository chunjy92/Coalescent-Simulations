# -*- coding: utf-8 -*-
"""
@author: CheYeol
"""

import numpy as np
from Bio import Phylo
from io import StringIO
from scipy.misc import comb
from scipy.stats import poisson

sample_size = 0
mu = 0
n = 0
mut_freq = {}


class Sample:  # Leaf of Tree
    def __init__(self, identity_count):
        self.identity = identity_count  # unique identity of each sample
        self.big_pivot = identity_count  # Conforms to usual visualization
        self.next = None  # links to its left neighbor child of its parent
        self.time = 0  # time to the previous coalescent event
        self.generation = 0  # each coalescent event represents a generation, beginning from the bottom of the tree
        self.mutations = 0  # mutations that occurred from the previous coalescent event

    def __repr__(self):
        return 'Sample {} with Mutations {:d}.'.format(self.identity, self.mutations)

    def is_sample(self):
        return True


class Ancestors(Sample):  # Internal Node of Tree
    def __init__(self, identity_count):
        super(self.__class__, self).__init__(identity_count)
        self.identity = 'A{}'.format(identity_count)
        self.generation = identity_count
        self.height = 0
        self.left = None  # left-most child
        self.right = None  # right-most child
        self.descendent_list = np.zeros(0)  # all samples below it
        self.children_list = np.zeros(0)  # all children directly below it

    def __repr__(self):
        return 'Ancestor {} with Mutations {:d}.'.format(self.identity, self.mutations)

    def is_sample(self):
        return False


def populate_coalescent_list(kingman_coalescent_list, bs_coalescent_list):
    for i in range(sample_size):
        kingman_coalescent_list[i] = Sample(i+1)
        bs_coalescent_list[i] = Sample(i+1)


def quicksort(children_list, first, last):
    '''
    sorts a children_list based on the value of big_pivot
    '''
    if first < last:
        splitpoint = partition(children_list, first, last)
        quicksort(children_list, first, splitpoint-1)
        quicksort(children_list, splitpoint+1, last)


def partition(children_list, first, last):
    lo, hi = first + 1, last
    piv = children_list[first].big_pivot
    while(True):
        while(lo <= hi and children_list[lo].big_pivot <= piv):
            lo += 1
        while(hi >= lo and children_list[hi].big_pivot >= piv):
            hi -= 1
        if hi < lo:
            break
        else:
            temp = children_list[lo]
            children_list[lo], children_list[hi] = children_list[hi], temp
    if hi == first:
        return hi
    part = children_list[first]
    children_list[first], children_list[hi] = children_list[hi], part
    return hi


def update_descendent_list(children_list):
    '''
    creates a descendent list by replacing samples in the children list
    with its own descendent list
    '''
    descendent_list = np.copy(children_list)
    i = 0
    while i < np.size(descendent_list):
        if not(descendent_list[i].is_sample()):
            # insert the internal node's own descendent list at the node's
            # index in the current descendent_list
            # -> since the node below the sample, its descd. list alrdy updated
            size = np.size(descendent_list[i].descendent_list)
            descendent_list = np.insert(descendent_list, i,
                                        descendent_list[i].descendent_list)
            i += size
            # remove the given internal node from the descendent list
            # -> we only want the samples, not the internal nodes
            descendent_list = np.delete(descendent_list, i)
        else:  # if sample
            i += 1  # move to the next on the descendent list
    return descendent_list


def k_F(n):
    '''
    kingman function
    '''
    return n*(n-1) / 2


def b_F(mn_rate, size):
    '''
    bolthausen-sznitman function
    '''
    total_rate = 0
    for i in range(0, size-1):
        m = i + 2
        i_rate = size / (m * (m-1))
        mn_rate[i] = i_rate
        total_rate += i_rate
    return mn_rate, total_rate


def heterozygosity_calculator(sample, k):
    '''
    helper method to calculate mean separation time
    '''
    b = comb(sample_size, 2)
    a = k * (sample_size-k)
    return a/b * sample.mutations


def variance_distribution_calculator(children_list):
    '''
    helper method to calculate variance distribution of variance of
    children's descendants
    '''
    num_desc = np.zeros_like(children_list)
    for i in range(len(num_desc)):
        if children_list[i].is_sample():
            num_desc[i] = 1
        else:
            num_desc[i] = len(children_list[i].descendent_list)
    return np.var(num_desc)


def coalesce_children(coalescent_list, identity_count, num_children=2):
    '''
    Given a number of children to be merged, perform a coalescent event
    that creates as a consequence a merge_sample or an ancestor
    '''
    # Create an Internal Node Representing a Coalescent Event
    merge_sample = Ancestors(identity_count)

    # The merge_sample's immediate children chosen
    children_list = np.random.choice(coalescent_list, num_children, replace=False)
    quicksort(children_list, 0, np.size(children_list)-1)  # sorted for visual ease
    coalescent_list = np.append(coalescent_list, merge_sample)
    return coalescent_list, merge_sample, children_list


def update_ancestor(ancestor, children_list, height):
    '''
    assign various attributes to the merged ancestor
    '''
    ancestor.children_list = children_list
    ancestor.descendent_list = update_descendent_list(children_list)
    ancestor.right = children_list[np.size(children_list) - 1]
    ancestor.big_pivot = ancestor.right.big_pivot
    ancestor.left = ancestor.children_list[0]
    ancestor.height = height
    return ancestor


def update_time(sample, ancestor, gen_time):
    for j in range(ancestor.generation-1, sample.generation-1, -1):
        sample.time += gen_time[j]
    return sample


def update_data(data_list, data_index, *data):
    for index, value in data:
        data_list[data_index][index] += value


def update_children(ancestor, data_list, data_index,
                    coalescent_list, children_list, gen_time):
    '''
    iterate through the ancestor's children list, setting up a tree, performing
    calculations simultaneously and storing results into data list
    '''
    temp_list = np.copy(children_list)
    height_list = np.zeros_like(temp_list)  # children's heights

    # children_index: index of changing children_list
    # heihgt_index: index of fixed children_list
    children_index, height_index = 0, 0

    # iteration through the children_list
    while children_index < np.size(temp_list):
        # current child under inspection
        current = temp_list[children_index]

        # add up the between-generaion time
        # and define the branche length in terms of mutations
        current = update_time(current, ancestor, gen_time)
        current.mutations = poisson.rvs(mu*current.time)
        try:
            mut_freq[str(current.mutations)] += 1
        except(KeyError):
            mut_freq[str(current.mutations)] = 1
        # data_list[data_index][0] += current.mutations

        # First Case : a Sample(Leaf)
        if current.is_sample():
            height_list[height_index] = current.mutations

            #data_list[data_index][4] += current.mutations
            heterozygosity = heterozygosity_calculator(current, 1)
            #data_list[data_index][2] += heterozygosity
            update_data(data_list, data_index,
                        *zip((0, 2, 5),(current.mutations, heterozygosity, current.mutations)))


        # Second Case : an Internal Node with Mutations == 0
        elif not(current.is_sample()) and current.mutations == 0:
            # Delete this Current Child from the Coalescent List
            cond = coalescent_list == temp_list[children_index]
            coalescent_list = np.delete(coalescent_list,
                                        int(np.where(cond)[0]))

            # Find the Max Height among the zero node's children
            temp_height_list = np.zeros_like(current.children_list)
            for h in range(0, len(temp_height_list)):
                try:
                    temp_height_list[h] = current.children_list[h].height + \
                                            current.children_list[h].mutations
                except (AttributeError):
                    temp_height_list[h] = current.children_list[h].mutations
            height_list[height_index] = np.amax(temp_height_list)

            # Replace this Current Child with Its children List
            temp_list = np.insert(temp_list, children_index, current.children_list)
            temp_list = np.delete(temp_list, children_index + np.size(current.children_list))

            # Create Linked List that Connects the Replacing children_list
            # with the original children on its left if it exists
            if children_index > 0:
                temp_list[children_index].next = temp_list[children_index-1]
            children_index += (np.size(current.children_list) - 1)

        # Third Case : an Internal Node with Mutations > 0
        else:
            height_list[height_index] = current.height + current.mutations
            #data_list[data_index][1] += 1
            heterozygosity = heterozygosity_calculator(current,
                                                       np.size(current.descendent_list))
            #data_list[data_index][2] += heterozygosity
            variance = variance_distribution_calculator(current.children_list)
            #data_list[data_index][5] += variance_distribution_calculator(current.children_list)
            update_data(data_list, data_index,
                        *zip((0, 1, 2, 6),(current.mutations, 1, heterozygosity, variance)))

        # Delete Current Child from the Coalescent List
        # Unless Deleted alrdy in the Second Case
        cond = coalescent_list == temp_list[children_index]
        if True in cond:
            coalescent_list = np.delete(coalescent_list, int(np.where(cond)[0]))

        # Link current child to its left
        if children_index > 0:
            current.next = temp_list[children_index-1]
        children_index += 1
        height_index += 1
    # Updating the Relevant Information
    ancestor = update_ancestor(ancestor, temp_list, np.amax(height_list))

    # If Most Recent Commont Ancestor
    if len(coalescent_list) == 1:
        #data_list[data_index][3] = ancestor.height

        highest_branch_length = 0
        for child in ancestor.children_list:
            highest_branch_length += child.mutations
        #if data_list[data_index][4] > 0:
        #    data_list[data_index][4] = highest_branch_length / data_list[data_index][4]
        variance = variance_distribution_calculator(ancestor.children_list)
        #data_list[data_index][5] += variance_distribution_calculator(ancestor.children_list)
        #print(mut_freq)
        max_mut_freq = int(max(mut_freq, key=mut_freq.get))
        #print("This max", max_mut_freq)
        update_data(data_list, data_index,
                    *zip((3, 4, 6, 7),
                         (ancestor.height, highest_branch_length, variance,
                         max_mut_freq)))
    return coalescent_list


def clear_mutation_frequency_dict():
    mut_freq.clear()


def kingman_coalescence(coalescent_list, *data):
    '''
    models the Kingman coalescent event
    '''
    gen_time = np.zeros(sample_size-1)  # coalescent time for each generation
    nth_coalescence = 0  # Starting from 0
    clear_mutation_frequency_dict()

    # Until reaching the Most Recent Common Ancestor
    while np.size(coalescent_list) > 1:
        # Time Calculation
        time = np.random.exponential(1/k_F(np.size(coalescent_list)))
        gen_time[nth_coalescence] = time
        nth_coalescence += 1

        # merged ancestor of the coalescent event and its children obtained
        consts = {'identity_count' : nth_coalescence}
        coalescent_list, ancestor, children_list = coalesce_children(coalescent_list,
                                                                     **consts )
        # update the tree using mutations as branch length
        # recording data along the way
        lists = {'coalescent_list' : coalescent_list, 'children_list' : children_list,
                 'gen_time' : gen_time}
        coalescent_list = update_children(ancestor, *data, **lists)
    return coalescent_list


def bs_coalescence(coalescent_list, *data):
    '''
    models the Bolthausen-Sznitman coalescent event
    '''
    gen_time = np.zeros(sample_size-1)  # coalescent time for each generation
    nth_coalescence = 0  # Starting from 0
    clear_mutation_frequency_dict()

    # Until reaching the Most Recent Common Ancestor
    while np.size(coalescent_list) > 1:
        # Time and Number of Children Calculation
        m_list = np.arange(2, np.size(coalescent_list)+1)
        mn_rate = np.zeros(np.size(coalescent_list)-1)
        bsf_rate = np.zeros(np.size(coalescent_list)-1)
        mn_rate, total_rate = b_F(mn_rate, np.size(coalescent_list))
        for j in range(0, np.size(mn_rate)):
            bsf_rate[j] = mn_rate[j] / total_rate
        num_children = np.random.choice(m_list, 1, replace=False, p=bsf_rate)
        time = np.random.exponential(1/total_rate)
        gen_time[nth_coalescence] = time
        nth_coalescence += 1

        # merged ancestor of the coalescent event and its children obtained
        consts = {'identity_count' : nth_coalescence, 'num_children' : num_children}
        coalescent_list, ancestor, children_list = coalesce_children(coalescent_list,
                                                                     **consts)
        # update the tree using mutations as branch length
        # recording data along the way
        lists = {'coalescent_list' : coalescent_list, 'children_list' : children_list,
                 'gen_time' : gen_time}
        coalescent_list = update_children(ancestor, *data, **lists)
    return coalescent_list


def traversal(sample):
    '''
    secondary method to establish newick form
    '''
    output = ''
    current = sample.right
    output = recur_traversal((output + '('), current)
    while current.next != sample.left:
        current = current.next
        output = recur_traversal(output + ', ', current)
    current = sample.left
    output = recur_traversal(output + ', ', current) + ')' + str(sample.identity)
    return output


def recur_traversal(output, sample):
    '''
    secondary method to establish newick form
    '''
    if sample.is_sample():
        output = output + str(sample.identity) + ':' + str(sample.mutations)
        return output
    current = sample.right
    output = recur_traversal((output + '('), current)
    while current.next != sample.left:
        current = current.next
        output = recur_traversal(output + ', ', current)
    current = sample.left
    output = recur_traversal((output + ', '), current)
    output = output + ')' + str(sample.identity) + ':' + str(sample.mutations)
    return output


def display_tree(ancestor):
    '''
    displays the Newick Format in String and the Phylo visualization
    '''
    newick = traversal(ancestor)
    tree = Phylo.read(StringIO(str(newick)), 'newick')
    Phylo.draw(tree)
    print(newick)
