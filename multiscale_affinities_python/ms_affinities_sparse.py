import numpy as np
import scipy.sparse as sps

# TODO replace with pure python blocking implementation
import nifty.tools as nt


# TODO this is extremely memory hungry
def compute_histograms(blocking, labels):
    histograms = []
    n_labels = int(labels.max() + 1)
    n_blocks = blocking.numberOfBlocks
    for block_id in range(n_blocks):

        if block_id % (n_blocks // 100) == 0:
            print(block_id, '/', n_blocks)

        block = blocking.getBlock(block_id)
        new_coordinate = tuple(blocking.blockGridPosition(block_id))
        roi = tuple(slice(begin, end) for begin, end in zip(block.begin, block.end))
        values, counts = np.unique(labels[roi], return_counts=True)
        h = sps.dok_matrix((n_labels, 1), dtype='float64')
        index = (values, np.zeros(len(values), dtype='uint32'))
        h[index] = counts
        histograms.append(h.tocsr())
    return histograms


def ms_single_scale_sparse(labels, block_shape):
    shape = labels.shape
    blocking = nt.blocking(roiBegin=[0, 0, 0],
                           roiEnd=list(shape),
                           blockShape=list(block_shape))

    # 1.) compute histograms from all blocks
    print("Computing histograms")
    histograms = compute_histograms(blocking, labels)

    # 2.) compute the new affinities
    print("Computing new affnities")
    new_shape = (3,) + tuple(blocking.blocksPerAxis)
    affs_out = np.zeros(new_shape, dtype='float32')

    # this can be trivially parallelised, but dunno how much this affords
    n_blocks = blocking.numberOfBlocks
    for block_id in range(n_blocks):

        if block_id % (n_blocks // 100) == 0:
            print(block_id, '/', n_blocks)

        block_histo = histograms[block_id]
        new_coordinate = tuple(blocking.blockGridPosition(block_id))
        # TODO only need this for normalization, which is currently broken
        # block = blocking.getBlock(block_id)
        # roi = tuple(slice(begin, end) for begin, end in zip(block.begin, block.end))

        # TODO it's easy to include long range here
        # we just need a way of going from block-axis positions to coordinates
        # iterate over the neighbors
        for axis in range(3):
            neighbor_id = blocking.getNeighborId(block_id, axis, True)
            if neighbor_id == -1:
                continue

            # TODO only need this for normalization, which is currently broken
            # neighbor_block = blocking.getBlock(neighbor_id)
            # nroi = tuple(slice(begin, end)
            #              for begin, end in zip(neighbor_block.begin, neighbor_block.end))

            neighbor_histo = histograms[neighbor_id]

            # get the normalization
            # FIXME something with this seems to be off
            # nshape = np.array(block.shape)
            # normalization = np.prod(block_shape) * np.prod(nshape)

            # FIXME normalization
            affinity = block_histo * neighbor_histo # / normalization
            aff_coordinate = (axis,) + new_coordinate
            affs_out[aff_coordinate] = affinity
    affs_out /= affs_out.max()
    return affs_out


# TODO support long ranges
def ms_sparse_implementation(labels, block_shapes):
    # TODO validate block-shapes
    out = [ms_single_scale_sparse(labels, bs) for bs in block_shapes]
    return out
