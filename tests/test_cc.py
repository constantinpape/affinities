import unittest
import numpy as np


# TODO check correctness for toy data
class CcTest(unittest.TestCase):

    def test_cc_2d(self):
        import affinities
        shape = (2, 100, 100)
        affs = np.random.rand(*shape)
        ccs, max_label = affinities.connected_components(affs, 0.5)
        self.assertEqual(ccs.shape, shape[1:])
        self.assertGreater(max_label, 10)
        self.assertEqual(max_label, ccs.max())

    def test_cc_3d(self):
        import affinities
        shape = (2, 100, 100, 100)
        affs = np.random.rand(*shape)
        ccs, max_label = affinities.connected_components(affs, 0.5)
        self.assertEqual(ccs.shape, shape[1:])
        self.assertGreater(max_label, 10)
        self.assertEqual(max_label, ccs.max())


if __name__ == '__main__':
    unittest.main()
