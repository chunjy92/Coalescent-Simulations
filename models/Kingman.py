# -*- coding: utf-8 -*-                     #
# ========================================= #
# Kingman Model                             #
# author      : Che Yeol (Jayeol) Chun      #
# last update : 03/17/2017                  #
# ========================================= #

import numpy as np

class Kingman:
    def __init__(self, n):
        self.k = self.__k_F(n)

    def __k_F(self, n):
        """
        computes the Kingman function
        @param n : Int   - current size of the coalescent list, i.e. number of samples available for coalescence
        @return  : Float - Kingman function result
        """
        return n*(n-1) / 2

    def __kingman_coalescence(self, sample_size, mu, coalescent_list,
                              *data, newick=False):
        """
        models the Kingman coalescence
        @param coalescent_list     : 1-d Array - holds samples
        @param data                : Tuple     - (data_list, data_index) -> refer to __update_data
        @param newick              : Bool      - refer to argument of src
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
            consts = {'identity_count': nth_coalescence}
            coalescent_list, ancestor, children_list = __coalesce_children(coalescent_list, **consts)

            # update the tree using mutations as branch length, recording data along the way
            lists = {'coalescent_list': coalescent_list, 'children_list': children_list, 'gen_time': gen_time}
            coalescent_list = __update_children(mu, ancestor, *data, **lists)
            if all_ancestors(coalescent_list):
                break