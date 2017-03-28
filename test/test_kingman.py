# -*- coding: utf-8 -*-                     #
# ========================================= #
# Kingman UnitTest                          #
# author      : Che Yeol (Jayeol) Chun      #
# last update : 03/27/2017                  #
# ========================================= #

import unittest

__author__ = 'Jayeol Chun'


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
