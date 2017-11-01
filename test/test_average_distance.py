import unittest

import numpy as np

from simulation.tree_utils import generate_trees


class MyTestCase(unittest.TestCase):
  def test_average_distance(self):
    """Average (shortest) distance between any two leaves"""
    np.random.seed(20)
    models = generate_trees(sample_sizes=[10, 30, 100, 300], num_iter=1000)

    self.assertEqual(len(models), 2)

    e = .02

    overall_avg = 0.
    overall_counter = 0
    for model in models:
      model_avg = 0.
      model_counter = 0
      for data in model.time_trees.values():

        for root in data:
          model_avg += root.pairwise_dist
          model_counter += 1
          overall_avg += root.pairwise_dist
          overall_counter += 1

      model_res = model_avg / model_counter
      self.assertAlmostEqual(model_res, 2., delta=e)
    overall_res =overall_avg / overall_counter
    self.assertAlmostEqual(overall_res, 2., delta=e)



if __name__ == '__main__':
  unittest.main()
