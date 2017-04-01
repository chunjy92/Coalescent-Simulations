# -*- coding: utf-8 -*-                     #
# ========================================= #
# Display Utils                             #
# author      : Che Yeol (Jayeol) Chun      #
# last update : 04/01/2017                  #
# ========================================= #

from typing import TypeVar, Tuple
import numpy as np
from Bio import Phylo
from io import StringIO
from models.structure import *


__author__ = 'Jayeol Chun'

T = TypeVar('T', Sample, Ancestor)

def display_params(args):
    print("\n****** Running with: ******")
    params = ['Sample Size', 'Mutation Rate', 'Number of Iterations']
    for param, arg in zip(params,args):
        print('\t{}: {}'.format(param, arg))


def display_tree(ancestor: Ancestor, verbose=False):
    """
    displays the Newick Format in string Newick format and its Phylo visualization
    @param ancestors : 1-d Array - root of the tree to be displayed
    """
    # for i in range(len(ancestors)):
    newick = _traversal(ancestor)
    tree = Phylo.read(StringIO(str(newick)), 'newick')
    Phylo.draw(tree)
    if verbose:
        print("\n*** Displaying Each Tree Results ***")
        print(newick)
        print(tree)

def display_stats(data_k, data_b, model_list, stat_list):
    """
    displays the cumulative statistics of all trees observed for Kingman and Bolthausen-Sznitman
    @param data_k     : 2-d Array - holds data extracted from Kingman trees
    @param data_b     : 2-d Array - holds data extracted from Bolthausen-Sznitman trees
    @param model_list : 1-d Array - provides the names of coalescent models
    @param stat_list  : 1-d Array - provides description of each statistics examined
    """
    m = len(stat_list)
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


def _traversal(sample: T) -> str:
    """
    iterates through the tree rooted at the sample recursively in pre-order, building up a Newick format
    @param sample  : Ancestor - root of the tree to be displayed
    @return output : String   - complete newick format
    """
    output = ''
    current = sample.right
    output = _recur_traversal((output + '('), current)
    while current.next != sample.left:
        current = current.next
        output = _recur_traversal(output + ', ', current)
    current = sample.left
    output = _recur_traversal(output + ', ', current) + ')' + str(sample.identity)
    return output


def _recur_traversal(output: str, sample: T) -> str:
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
    output = _recur_traversal((output + '('), current)
    while current.next != sample.left:
        current = current.next
        output = _recur_traversal(output + ', ', current)
    current = sample.left
    output = _recur_traversal((output + ', '), current)
    output = output + ')' + str(sample.identity) + ':' + str(sample.mutations)
    return output