# -*- coding: utf-8 -*-                     #
# ========================================= #
# Quicksort UnitTest                        #
# author      : Che Yeol (Jayeol) Chun      #
# last update : 04/01/2017                  #
# ========================================= #

import unittest

from random import shuffle, randint
from models.structure import *
from utils.sorting import quicksort

__author__ = 'Jayeol Chun'


class MyTestCase(unittest.TestCase):
    def test_quicksort(self):
        rands = [randint(1, 100) for _ in range(200)]

        temp = [Sample(rand) for rand in rands]
        temp.sort(key=lambda x: x.big_pivot)

        samples = [Sample(rand) for rand in rands]
        shuffle(samples)
        quicksort(samples, 0, len(samples)-1)

        self.assertListEqual([sample.big_pivot for sample in temp], [sample.big_pivot for sample in samples])


if __name__ == '__main__':
    unittest.main()
