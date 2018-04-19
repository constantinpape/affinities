#pragma once
#include "affinities/multiscale_affinities.hxx"

namespace affinities {

    template<class LABEL_ARRAY, class AFFS_ARRAY, class MASK_ARRAY>
    void computeFullscaleMultiscaleAffinities(const xt::xexpression<LABEL_ARRAY> & labelsExp,
                                              const std::vector<int> & blockShape,
                                              xt::xexpression<AFFS_ARRAY> & affsExp,
                                              xt::xexpression<MASK_ARRAY> & maskExp,
                                              const bool haveIgnoreLabel=false,
                                              const uint64_t ignoreLabel=0) {
        const auto & labels = labelsExp.derived_cast();
        auto & affs = affsExp.derived_cast();
        auto & mask = maskExp.derived_cast();

        // get the shape
        Coordinate shape;
        for(unsigned d = 0; d < 3; ++d) {
            shape[d] = mask.shape()[d];
        }

        // compute the half block shape (this assumes that block shapes are odd !)
        Coordinate blockLens;
        for(unsigned d = 0; d < 3; ++d) {
            blockLens[d] = blockShape[d] / 2;
        }

        // iterate over all pixels
        for(int64_t i = 0; i < shape[0]; ++i) {
            for(int64_t j = 0; j < shape[1]; ++j) {
                for(int64_t k = 0; k < shape[2]; ++k) {
                    // declare histograms
                    Histogram hist, ngb;

                    // define central block
                    // TODO how to handle out of range
                    // clip (as done now)
                    // or mask
                    Coordinate coordinate = {i, j, k};
                    Coordinate blockBegin, blockEnd;
                    for(unsigned d = 0; d < 3; ++d) {
                        blockBegin[d] = coordinate[d] - blockLens[d];
                        blockBegin[d] = std::max(blockBegin[d], 0L);
                        blockEnd[d] = coordinate[d] + blockLens[d];
                        blockEnd[d] = std::min(blockEnd[d], shape[d]);
                    }

                    // compute central histogram
                    size_t blockSize;
                    if(haveIgnoreLabel){
                        blockSize = computeHistogramWithIgnoreLabel(labels, blockBegin, blockEnd,
                                                                    hist, ignoreLabel);
                    }
                    else {
                        blockSize = computeHistogram(labels, blockBegin, blockEnd, hist);
                    }

                    // if the block size is zero (due to masking), mask these affinities and continue
                    if(blockSize == 0) {
                        mask(0, i, j, k) = 0;
                        mask(1, i, j, k) = 0;
                        mask(2, i, j, k) = 0;
                        continue;
                    }

                    for(unsigned d = 0; d < 3; ++d) {
                        ngb.clear();
                        // compute affinity to i-neighbor block
                        Coordinate ngbBegin = blockBegin;
                        Coordinate ngbEnd = blockEnd;

                        ngbBegin[d] -= blockShape[d];
                        ngbEnd[d] -= blockShape[d];

                        if(ngbBegin[d] < 0) {
                            mask(d, i, j, k) = 0;
                            continue;
                        }

                        // compute neighboring histogram
                        size_t ngbSize;
                        if(haveIgnoreLabel){
                            ngbSize = computeHistogramWithIgnoreLabel(labels, blockBegin, blockEnd,
                                                                      ngb, ignoreLabel);
                        }
                        else {
                            ngbSize = computeHistogram(labels, blockBegin, blockEnd, ngb);
                        }

                        // if the block size is zero (due to masking), mask these affinities and continue
                        if(ngbSize == 0) {
                            mask(d, i, j, k) = 0;
                            continue;
                        }

                        // compute the affinity
                        const double norm = blockSize * ngbSize;
                        affs(d, i, j, k) = computeSingleAffinity(hist, ngb, norm);
                        mask(d, i, j, k) = 1;
                    }
                }
            }
        }
    }

}
