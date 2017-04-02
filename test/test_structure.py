# -*- coding: utf-8 -*-                     #
# ========================================= #
# Structure Unittest                        #
# author      : Che Yeol (Jayeol) Chun      #
# last update : 04/01/2017                  #
# ========================================= #

import unittest

from models.structure import *
from models.update import *

__author__ = 'Jayeol Chun'


class MyTestCase(unittest.TestCase):
    def test_structure(self):
        root = Ancestor(10)
        mid_ancestor = Ancestor(9)
        low_ancestor_1 = Ancestor(8)
        low_ancestor_2 = Ancestor(7)
        sample_1 = Sample(1)
        sample_2 = Sample(2)
        sample_3 = Sample(3)
        sample_4 = Sample(4)
        sample_5 = Sample(5)
        samples = [sample_1, sample_2, sample_3, sample_4, sample_5]

        self.assertTrue(sample_1.is_sample())
        self.assertFalse(root.is_sample())

        update_ancestor(low_ancestor_1,[sample_1, sample_2] )
        update_ancestor(low_ancestor_2, [sample_3, sample_4, sample_5])
        update_ancestor(mid_ancestor, [low_ancestor_1, low_ancestor_2])
        update_ancestor(root, [mid_ancestor])

        # root level
        self.assertEqual(root.children_list[0], mid_ancestor) # direct child
        self.assertListEqual(root.descendent_list, samples) # descendents are only Sample objs
        self.assertEqual(root.big_pivot, mid_ancestor.big_pivot)

        # mid level
        self.assertListEqual(mid_ancestor.children_list, [low_ancestor_1, low_ancestor_2])
        self.assertListEqual(mid_ancestor.descendent_list, samples) # descendents are only Sample objs
        self.assertGreater(low_ancestor_2.big_pivot, low_ancestor_1.big_pivot) # 5 vs 2
        self.assertEqual(mid_ancestor.right, low_ancestor_2) # bigger big_pivot becomes right
        self.assertEqual(mid_ancestor.left, low_ancestor_1) # converse

        # low level
        self.assertListEqual(low_ancestor_1.children_list, low_ancestor_1.descendent_list)
        self.assertEqual(low_ancestor_1.right, sample_2)
        self.assertEqual(low_ancestor_1.left, sample_1)
        self.assertListEqual(low_ancestor_2.children_list, low_ancestor_2.descendent_list)
        self.assertEqual(low_ancestor_2.right, sample_5)
        self.assertEqual(low_ancestor_2.left, sample_3)

if __name__ == '__main__':
    unittest.main()
