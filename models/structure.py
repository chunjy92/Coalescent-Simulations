# -*- coding: utf-8 -*-                     #
# ========================================= #
# Coalescent Simulations Tree Structure     #
# author      : Che Yeol (Jayeol) Chun      #
# last update : 03/17/2017                  #
# ========================================= #

import numpy as np

########################### Tree Node Objects ###########################

class Sample:  # Leaf of Tree
    def __init__(self, identity_count):
        """
        @param identity_count: Int - unique ID number to distinguish this sample from the rest
        """
        self.identity = str(identity_count) # unique identity of each sample
        self.big_pivot = identity_count     # used as key values in quicksort - conforms to usual visualization by placing nodes with bigger pivots
                                            # to the right of the children_list and descendant_list
        self.next = None                    # links to its left neighbor child of its parent
        self.time = 0                       # time to the previous coalescent event
        self.generation = 0                 # each coalescent event represents a generation, beginning from the bottom of the tree
        self.mutations = 0                  # mutations that occurred from the previous coalescent event

    def __repr__(self):
        return 'Sample {} with Mutations {}.'.format(self.identity, self.mutations)

    def is_sample(self):
        return 'A' not in self.identity

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
        return 'Ancestor {} with Mutations {}.'.format(self.identity, self.mutations)