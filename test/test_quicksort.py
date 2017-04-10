# -*- coding: utf-8 -*-

import unittest
from random import shuffle, randint
from models import Sample
from models.utils import quicksort

__author__ = 'Jayeol Chun'


class QuicksortTest(unittest.TestCase):
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
