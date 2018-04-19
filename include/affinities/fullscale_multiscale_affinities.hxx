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
            shape[d] = labels.shape()[d];
        }

        // get the strides
        const Coordinate strides = {shape[1] * shape[2], shape[2], 1};

        // compute the half block shape
        // this assumes that block shapes are odd,
        // to have the pixel in the middle of the block.
        // otherwise the pixel will be misaligned
        Coordinate blockLens;
        for(unsigned d = 0; d < 3; ++d) {
            blockLens[d] = blockShape[d] / 2;
        }

        // compute the histograms
        const size_t numberOfPixels = std::accumulate(shape.begin(),
                                                      shape.end(), 1,
                                                      std::multiplies<size_t>());
        HistogramStorage histograms(numberOfPixels);
        std::vector<size_t> blockSizes(numberOfPixels, 0);

        // compute histogram for each pixel
        Coordinate blockBegin, blockEnd, coord;
        for(int64_t i = 0; i < shape[0]; ++i) {
            for(int64_t j = 0; j < shape[1]; ++j) {
                for(int64_t k = 0; k < shape[2]; ++k) {

                    // get flat pixel index and coordinate
                    coord = {i, j, k};
                    const size_t pixId = i * strides[0] + j * strides[1] + k * strides[2];

                    // calculate the block around this pixel
                    bool outOfRange = false;
                    for(unsigned d = 0; d < 3; ++d) {
                        blockBegin[d] = coord[d] - blockLens[d];
                        // we need to add one to the block end, becuase it is
                        // exclusive (begin is inclusive !)
                        blockEnd[d] = coord[d] + blockLens[d] + 1;
                        // check if this block is in range
                        if(blockBegin[d] < 0 || blockEnd[d] > shape[d]) {
                            outOfRange = true;
                            break;
                        }
                    }

                    // don't compute affinity and leave block size at 0 if the block
                    // is out of range
                    if(outOfRange) {
                        continue;
                    }

                    // compute central histogram
                    size_t & blockSize = blockSizes[pixId];
                    if(haveIgnoreLabel){
                        blockSize = computeHistogramWithIgnoreLabel(labels, blockBegin, blockEnd,
                                                                    histograms[pixId], ignoreLabel);
                    }
                    else {
                        blockSize = computeHistogram(labels, blockBegin, blockEnd, histograms[pixId]);
                    }
                }
            }
        }

        // compute affinities for each pixel
        for(int64_t i = 0; i < shape[0]; ++i) {
            for(int64_t j = 0; j < shape[1]; ++j) {
                for(int64_t k = 0; k < shape[2]; ++k) {

                    // get flat pixel index and this histogram
                    const size_t pixId = i * strides[0] + j * strides[1] + k * strides[2];
                    const size_t blockSize = blockSizes[pixId];

                    // check if this pixel is valid (we use block size as a proxy)
                    if(blockSize == 0) {
                        mask(0, i, j, k) = 0;
                        mask(1, i, j, k) = 0;
                        mask(2, i, j, k) = 0;
                        continue;
                    }
                    const auto & hist = histograms[pixId];
                    const Coordinate coord = {i, j, k};

                    // compute affinity to neighbor blocks
                    for(unsigned d = 0; d < 3; ++d) {

                        // get the neighbor coordinate
                        Coordinate ngbCoord = coord;
                        ngbCoord[d] -= 1;

                        // check if the neighbor coordinate is valid
                        if(ngbCoord[d] < 0) {
                            mask(d, i, j, k) = 0;
                            continue;
                        }

                        // get the neighbor id and check if it is masked
                        const size_t ngbId = ngbCoord[0] * strides[0] + ngbCoord[1] * strides[1] + ngbCoord[2] * strides[2];
                        const size_t ngbSize = blockSizes[ngbId];
                        if(ngbSize == 0) {
                            mask(d, i, j, k) = 0;
                            continue;
                        }
                        const auto & ngbHist = histograms[ngbId];

                        // compute the affinity
                        const double norm = blockSize * ngbSize;
                        affs(d, i, j, k) = computeSingleAffinity(hist, ngbHist, norm);
                        mask(d, i, j, k) = 1;
                    }
                }
            }
        }
    }

}
