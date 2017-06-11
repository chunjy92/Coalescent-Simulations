import unittest
import numpy as np
from tree_utils import generate_trees, heterozygosity_calculator

class MyTestCase(unittest.TestCase):
    def test_average_distance(self):
        """Average (shortest) distance between any two leaves"""
        np.random.seed(20)

        # sample_sizes = range(10, 15, 2)
        # num_iter = 1
        #
        # trees
        data = generate_trees(sample_sizes=[10], num_iter=1)

        self.assertEqual(len(data), 2)

        for modelName, data in data.items():
            print("Model:", modelName)
            for sample_size, tree_list in data.items():
                children = tree_list[0].descendent_list
                print(children)
                het = 0
                for child in children:
                    het += heterozygosity_calculator(sample_size, len(children))
                het /= len(children)
                print("HET:", het)
                # print(het)
            # print(sample_size, trees)
            # tree = trees[sample_size]
            # children = tree.descendent_list
            # print(children)

        # for tree in trees.values():
        #     # children = tree[0]
        #     # print(children)
        #     # print(type(children))
        #     print(tree, type(tree))
        #     print(tree.descendent_list)
            # leaves = tree.descendant_list
            # print(len(leaves))

if __name__ == '__main__':
    unittest.main()
