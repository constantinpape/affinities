import unittest
import numpy as np


# TODO check correctness for toy data
# TODO check ignore data
class MalisTest(unittest.TestCase):

    def test_malis_2d(self):
        import affinities
        shape = (100, 100)
        labels = np.random.randint(0, 100, size=shape)
        offsets = [[-1, 0], [0, -1]]
        affs, _ = affinities.compute_affinities(labels, offsets)
        affs += 0.1 * np.random.randn(*affs.shape)
        loss, grads = affinities.compute_malis_2d(affs, labels, offsets)
        self.assertEqual(grads.shape, affs.shape)
        # FIXME this fails
        self.assertNotEqual(loss, 0)
        self.assertFalse(np.allclose(grads, 0))

    def test_malis_3d(self):
        import affinities
        shape = (100, 100, 100)
        labels = np.random.randint(0, 1000, size=shape)
        offsets = [[-1, 0], [0, -1]]
        affs, _ = affinities.compute_affinities(labels, offsets)
        affs += 0.1 * np.random.randn(*affs.shape)
        loss, grads = affinities.compute_malis_3d(affs, labels, offsets)
        self.assertEqual(grads.shape, affs.shape)
        # FIXME this fails
        self.assertNotEqual(loss, 0)
        self.assertFalse(np.allclose(grads, 0))


if __name__ == '__main__':
    unittest.main()
