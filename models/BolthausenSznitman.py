# -*- coding: utf-8 -*-                     #
# ========================================= #
# Bolthausen-Sznitman Model                 #
# author      : Che Yeol (Jayeol) Chun      #
# last update : 03/17/2017                  #
# ========================================= #

import numpy as np

class BolthausenSznitman:
    def __init__(self):
        pass

    def __b_F(self, mn_rate, n):
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

    def __bs_coalescence(self, sample_size, mu, coalescent_list,
                         *data, newick=False):
        """
        models the Bolthausen-Sznitman coalescence
        @param sample_size     : Int       - refer to argument of src
        @param mu              : Float     - refer to argument of src
        @param coalescent_list : 1-d Array - holds samples
        @param data            : Tuple     - (data_list, data_index) -> refer to __update_data
        @param newick          : Bool      - refer to argument of src
        """
        gen_time = np.zeros(sample_size - 1)  # coalescent time for each generation
        nth_coalescence = 0  # Starting from 0

        # Until reaching the Most Recent Common Ancestor
        while np.size(coalescent_list) > 1:
            # Time and Number of Children Calculation
            m_list = np.arange(2, np.size(coalescent_list) + 1)
            mn_rate = np.zeros(np.size(coalescent_list) - 1)
            bsf_rate = np.zeros(np.size(coalescent_list) - 1)
            mn_rate, total_rate = __b_F(mn_rate, np.size(coalescent_list))
            for j in range(0, np.size(mn_rate)):
                bsf_rate[j] = mn_rate[j] / total_rate
            num_children = np.random.choice(m_list, 1, replace=False, p=bsf_rate)
            time = np.random.exponential(1 / total_rate)
            gen_time[nth_coalescence] = time
            nth_coalescence += 1

            # merged ancestor of the coalescent event and its children obtained
            consts = {'identity_count': nth_coalescence, 'num_children': num_children}
            coalescent_list, ancestor, children_list = __coalesce_children(coalescent_list, **consts)

            # update the tree using mutations as branch length, recording data along the way
            lists = {'coalescent_list': coalescent_list, 'children_list': children_list, 'gen_time': gen_time}

            coalescent_list = __update_children(mu, ancestor, *data, **lists)
            if all_ancestors(coalescent_list):
                break



