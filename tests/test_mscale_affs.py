import unittest
import numpy as np


class MultiscaleAffinitiesTest(unittest.TestCase):

    def test_ms_affs(self):
        import multiscale_affinities
        shape = (100, 100, 100)
        labels = np.random.randint(0, 100, size=shape)
        block_shapes = [[2, 2, 2], [10, 10, 10], [5, 5, 1]]
        for block_shape in block_shapes:
            affs = multiscale_affinities.compute_multiscale_affinities(labels, block_shape)
            expected_shape = (3,) + tuple(sh // bs for sh, bs in zip(shape, block_shape))
            self.assertEqual(affs.shape, expected_shape)

    def test_ms_affs_ignore(self):
        import multiscale_affinities
        shape = (100, 100, 100)
        labels = np.random.randint(0, 100, size=shape)
        block_shapes = [[2, 2, 2], [10, 10, 10], [5, 5, 1]]
        for block_shape in block_shapes:
            affs = multiscale_affinities.compute_multiscale_affinities(labels, block_shape, True, 0)
            expected_shape = (3,) + tuple(sh // bs for sh, bs in zip(shape, block_shape))
            self.assertEqual(affs.shape, expected_shape)


if __name__ == '__main__':
    unittest.main()
