# -*- coding: utf-8 -*-

import unittest
import numpy as np
from models import Sample, Kingman, BolthausenSznitman

__author__ = 'Jayeol Chun'


class ModelsTest(unittest.TestCase):
  def test_kingman(self):
    np.random.seed(20)

    # param setting
    sample_size = 6
    mu = 0.5
    model = Kingman(sample_size, mu)
    s1 = Sample(1)
    s2 = Sample(2)
    s3 = Sample(3)
    s4 = Sample(4)
    s5 = Sample(5)
    s6 = Sample(6)
    candidates = [s1, s2, s3, s4, s5, s6]
    data = np.zeros((1, 1))

    root = model.coalesce(candidates, (0, data), verbose=False)
    self.assertEqual(root.identity, 'K5')
    self.assertFalse(root.is_sample())

    # print(root.children_list)
    for child in root.children_list:
      if not child.is_sample():
        self.assertGreater(child.mutations, 0)

    # Most Important : Make sure generation times add up.
    # Calculation done internally, but can be confirmed.
    # turn verbose=True for model.coalesce above
    self.assertGreater(s3.time, s1.time)
    self.assertGreater(s1.time, s2.time)
    self.assertEqual(s2.time, s5.time)
    self.assertGreater(s2.time, s4.time)
    self.assertEqual(s4.time, s6.time)


    # display_tree(root, verbose=True)


  def test_bolthausen_sznitman(self):
    np.random.seed(20)

    # param setting
    sample_size = 6
    mu = 0.5
    model = BolthausenSznitman(sample_size, mu)
    s1 = Sample(1)
    s2 = Sample(2)
    s3 = Sample(3)
    s4 = Sample(4)
    s5 = Sample(5)
    s6 = Sample(6)
    candidates = [s1, s2, s3, s4, s5, s6]
    data = np.zeros((1, 1))

    root = model.coalesce(candidates, (0, data), verbose=False)
    self.assertEqual(root.identity, 'B3')
    self.assertFalse(root.is_sample())
    # A2: empty node, s3 becomes the direct children as aresult
    self.assertCountEqual(root.children_list, [s1, s2, s5, s3, root.children_list[4]])
    self.assertEqual(s1.time, s2.time)
    self.assertEqual(s2.time, s5.time)
    self.assertGreater(s5.time, s3.time)
    self.assertGreater(s3.time, s4.time)
    self.assertEqual(s4.time, s6.time)

    # display_tree(root, True)


if __name__ == '__main__':
  unittest.main()
