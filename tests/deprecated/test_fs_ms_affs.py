import unittest
import numpy as np


class FullscaleMultiscaleAffinitiesTest(unittest.TestCase):

    def test_fs_ms_affs(self):
        import affinities
        shape = (100, 100, 100)
        labels = np.random.randint(0, 100, size=shape)
        block_shapes = [[2, 2, 2], [10, 10, 10], [5, 5, 1]]
        expected_shape = (3,) + shape
        for block_shape in block_shapes:
            affs, mask = affinities.compute_fullscale_multiscale_affinities(labels, block_shape)
            self.assertEqual(affs.shape, expected_shape)
            self.assertEqual(mask.shape, expected_shape)
            self.assertNotEqual(np.sum(affs == 0), 0)
            self.assertNotEqual(np.sum(mask == 0), 0)
            # print(np.sum(affs))

    def test_fs_ms_affs_ignore(self):
        import affinities
        shape = (100, 100, 100)
        labels = np.random.randint(0, 100, size=shape)
        block_shapes = [[2, 2, 2], [10, 10, 10], [5, 5, 1]]
        expected_shape = (3,) + shape
        for block_shape in block_shapes:
            affs, mask = affinities.compute_fullscale_multiscale_affinities(labels,
                                                                            block_shape,
                                                                            True, 0)
            self.assertEqual(affs.shape, expected_shape)
            self.assertEqual(mask.shape, expected_shape)
            self.assertNotEqual(np.sum(affs == 0), 0)
            self.assertNotEqual(np.sum(mask == 0), 0)
            # print(np.sum(affs))


if __name__ == '__main__':
    unittest.main()
