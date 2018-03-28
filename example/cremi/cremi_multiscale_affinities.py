import numpy as np
import h5py
import vigra
import time

from multiscale_affinities_python import ms_single_scale_dense
from multiscale_affinities import compute_multiscale_affinities
from cremi_tools.viewer.volumina import view


def ms_affs_cremi():

    path = '/home/papec/Work/neurodata_hdd/cremi/sample_A_20160501.hdf'
    bb = np.s_[:50, :512, :512]
    with h5py.File(path, 'r') as f:
        gt = f['volumes/labels/neuron_ids'][bb]

    # sampling_factors = [3, 9, 9]
    sampling_factors = [1, 3, 3]

    # t0 = time.time()
    # ms_affs_py = ms_single_scale_dense(gt, sampling_factors)
    # print('Ms affs python in', time.time() - t0)

    t1 = time.time()
    ms_affs = compute_multiscale_affinities(gt, sampling_factors)
    print("Ms affs in", time.time() - t1)

    # assert ms_affs.shape == ms_affs_py.shape, "%s, %s" % (str(ms_affs.shape), str(ms_affs_py.shape))

    with h5py.File(path, 'r') as f:
        raw = f['volumes/raw'][bb].astype('float32')

    new_shape = ms_affs.shape[1:]
    raw = vigra.sampling.resize(raw, new_shape)
    # view([raw, ms_affs.transpose((1, 2, 3, 0)), ms_affs_py.transpose((1, 2, 3, 0))],
    #      ['raw', 'ms-affs-cpp', 'ms-affs-py'])
    view([raw, ms_affs.transpose((1, 2, 3, 0))],
         ['raw', 'ms-affs-cpp'])


if __name__ == '__main__':
    ms_affs_cremi()
