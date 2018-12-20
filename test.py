import unittest

import numpy as np

import main



class TestCyclops(unittest.TestCase):

  def test_solve_matrix(self):
    # This matrix used to fail because of the very small coefficient.
    MATRIX = np.array([[1.73759294e-17, -.428571429, .214285714],
                       [.428571429, 0, 0],
                       [.0666666667, .421122995, -.416666667],
                       [-.214285714, 0, 0],
                       [-.666666667, -.333333333, .250000000],
                       [-1, -1, -.333333333], [-1, -1, -1], [-1, -1, -1]])
    strategy, value = main.solve_matrix_game(MATRIX)
    self.assertGreaterEqual(value, -1)
    self.assertLessEqual(value, 1)


if __name__ == '__main__':
  unittest.main()
