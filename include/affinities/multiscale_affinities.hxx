#pragma once
#include <unordered_map>
#include <vector>
#include "xtensor/xtensor.hpp"


namespace affinities {

    // typedefs
    typedef std::array<int64_t, 2> Coordinate2D;
    typedef std::array<int64_t, 3> Coordinate3D;
    typedef std::unordered_map<uint64_t, size_t> Histogram;
    typedef std::vector<Histogram> HistogramStorage;


    template<class LABELS>
    inline size_t computeHistogram(const LABELS & labels,
                                   const Coordinate3D & blockBegin,
                                   const Coordinate3D & blockEnd,
                                   Histogram & out) {
        size_t nPixels = 0;
        for(int64_t z = blockBegin[0]; z < blockEnd[0]; ++z) {
            for(int64_t y = blockBegin[1]; y < blockEnd[1]; ++y) {
                for(int64_t x = blockBegin[2]; x < blockEnd[2]; ++x) {
                    //
                    const uint64_t label = labels(z, y, x);
                    auto outIt = out.find(label);
                    if(outIt == out.end()) {
                        out.insert(outIt, std::make_pair(label, 1));
                    } else {
                        ++outIt->second;
                    }
                    ++nPixels;
                }
            }
        }
        return nPixels;
    }


    template<class LABELS>
    inline size_t computeHistogram(const LABELS & labels,
                                   const Coordinate2D & blockBegin,
                                   const Coordinate2D & blockEnd,
                                   Histogram & out) {
        size_t nPixels = 0;
        for(int64_t x = blockBegin[0]; x < blockEnd[0]; ++x) {
            for(int64_t y = blockBegin[1]; y < blockEnd[1]; ++y) {
                //
                const uint64_t label = labels(x, y);
                auto outIt = out.find(label);
                if(outIt == out.end()) {
                    out.insert(outIt, std::make_pair(label, 1));
                } else {
                    ++outIt->second;
                }
                ++nPixels;
            }
        }
        return nPixels;
    }


    template<class LABELS>
    inline size_t computeHistogramWithIgnoreLabel(const LABELS & labels,
                                                  const Coordinate3D & blockBegin,
                                                  const Coordinate3D & blockEnd,
                                                  Histogram & out,
                                                  const uint64_t ignoreLabel) {
        size_t nPixels = 0;
        for(int64_t z = blockBegin[0]; z < blockEnd[0]; ++z) {
            for(int64_t y = blockBegin[1]; y < blockEnd[1]; ++y) {
                for(int64_t x = blockBegin[2]; x < blockEnd[2]; ++x) {
                    //
                    const uint64_t label = labels(z, y, x);
                    if(label == ignoreLabel) {
                        continue;
                    }

                    auto outIt = out.find(label);
                    if(outIt == out.end()) {
                        out.insert(outIt, std::make_pair(label, 1));
                    } else {
                        ++outIt->second;
                    }
                    ++nPixels;
                }
            }
        }
        return nPixels;
    }


    template<class LABELS>
    inline size_t computeHistogramWithIgnoreLabel(const LABELS & labels,
                                                  const Coordinate2D & blockBegin,
                                                  const Coordinate2D & blockEnd,
                                                  Histogram & out,
                                                  const uint64_t ignoreLabel) {
        size_t nPixels = 0;
        for(int64_t x = blockBegin[0]; x < blockEnd[0]; ++x) {
            for(int64_t y = blockBegin[1]; y < blockEnd[1]; ++y) {
                //
                const uint64_t label = labels(x, y);
                if(label == ignoreLabel) {
                    continue;
                }

                auto outIt = out.find(label);
                if(outIt == out.end()) {
                    out.insert(outIt, std::make_pair(label, 1));
                } else {
                    ++outIt->second;
                }
                ++nPixels;
            }
        }
        return nPixels;
    }


    inline double computeSingleAffinity(const Histogram & histo, const Histogram & ngbHisto,
                                        const double norm) {
        double aff = 0.;
        for(auto it = histo.begin(); it != histo.end(); ++it){
            auto ngbIt = ngbHisto.find(it->first);
            if (ngbIt != ngbHisto.end()) {
                aff += it->second * ngbIt->second;
            }
        }
        return aff / norm;
    }


    inline size_t getBlockIndex(const size_t i, const size_t j, const size_t k,
                                const Coordinate3D & blockStrides) {
        return blockStrides[0] * i + blockStrides[1] * j + blockStrides[2] * k;
    }


    inline size_t getBlockIndex(const size_t i, const size_t j,
                                const Coordinate2D & blockStrides) {
        return blockStrides[0] * i + blockStrides[1] * j;
    }


    template<class LABEL_ARRAY, class AFFS_ARRAY, class MASK_ARRAY>
    void computeMultiscaleAffinities2D(const xt::xexpression<LABEL_ARRAY> & labelsExp,
                                       const std::vector<int> & blockShape,
                                       xt::xexpression<AFFS_ARRAY> & affsExp,
                                       xt::xexpression<MASK_ARRAY> & maskExp,
                                       const bool haveIgnoreLabel=false,
                                       const uint64_t ignoreLabel=0) {

        const auto & labels = labelsExp.derived_cast();
        auto & affs = affsExp.derived_cast();
        auto & mask = maskExp.derived_cast();

        //
        // compute the block sizes and number of blocks
        //
        Coordinate2D shape;
        Coordinate2D blocksPerAxis;
        for(unsigned d = 0; d < 2; ++d) {
            blocksPerAxis[d] = affs.shape()[d + 1];
            shape[d] = labels.shape()[d];
        }
        const size_t numberOfBlocks = std::accumulate(blocksPerAxis.begin(),
                                                      blocksPerAxis.end(), 1,
                                                      std::multiplies<size_t>());
        const Coordinate2D blockStrides = {blocksPerAxis[1], 1};

        //
        // compute the histograms
        //
        HistogramStorage histograms(numberOfBlocks);
        std::vector<size_t> blockSizes(numberOfBlocks, 1);
        for(int64_t i = 0; i < blocksPerAxis[0]; ++i) {
            for(int64_t j = 0; j < blocksPerAxis[1]; ++j) {

                const size_t blockId = getBlockIndex(i, j, blockStrides);
                Coordinate2D blockBegin = {i * blockShape[0], j * blockShape[1]};
                Coordinate2D blockEnd = {std::min((i + 1) * blockShape[0], shape[0]),
                                         std::min((j + 1) * blockShape[1], shape[1])};

                size_t & blockSize = blockSizes[blockId];
                if(haveIgnoreLabel){
                    blockSize = computeHistogramWithIgnoreLabel(labels, blockBegin, blockEnd,
                                                                histograms[blockId],
                                                                ignoreLabel);
                }
                else {
                    blockSize = computeHistogram(labels, blockBegin,
                                                 blockEnd, histograms[blockId]);
                }
            }
        }

        //
        // compute the affinties
        //
        for(int64_t i = 0; i < blocksPerAxis[0]; ++i) {
            for(int64_t j = 0; j < blocksPerAxis[1]; ++j) {

                const size_t blockId = getBlockIndex(i, j, blockStrides);
                const auto & histo = histograms[blockId];
                const size_t blockSize = blockSizes[blockId];

                if(i > 0) {
                    const size_t ngbId = getBlockIndex(i - 1, j, blockStrides);
                    const auto & ngbHisto = histograms[ngbId];
                    const double norm = blockSize * blockSizes[ngbId];
                    if(norm > 0) {
                        affs(0, i, j) = computeSingleAffinity(histo, ngbHisto, norm);
                        mask(0, i, j) = 1;
                    } else {
                        mask(0, i, j) = 0;
                    }
                } else {
                    mask(0, i, j) = 0;
                }

                if(j > 0) {
                    const size_t ngbId = getBlockIndex(i, j - 1, blockStrides);
                    const auto & ngbHisto = histograms[ngbId];
                    const double norm = blockSize * blockSizes[ngbId];
                    if(norm > 0) {
                        affs(1, i, j) = computeSingleAffinity(histo, ngbHisto, norm);
                        mask(1, i, j) = 1;
                    } else {
                        mask(1, i, j) = 0;
                    }
                } else {
                    mask(1, i, j) = 0;
                }
            }
        }
    }


    template<class LABEL_ARRAY, class AFFS_ARRAY, class MASK_ARRAY>
    void computeMultiscaleAffinities3D(const xt::xexpression<LABEL_ARRAY> & labelsExp,
                                       const std::vector<int> & blockShape,
                                       xt::xexpression<AFFS_ARRAY> & affsExp,
                                       xt::xexpression<MASK_ARRAY> & maskExp,
                                       const bool haveIgnoreLabel=false,
                                       const uint64_t ignoreLabel=0) {

        const auto & labels = labelsExp.derived_cast();
        auto & affs = affsExp.derived_cast();
        auto & mask = maskExp.derived_cast();

        //
        // compute the block sizes and number of blocks
        //
        Coordinate3D shape;
        Coordinate3D blocksPerAxis;
        for(unsigned d = 0; d < 3; ++d) {
            blocksPerAxis[d] = affs.shape()[d + 1];
            shape[d] = labels.shape()[d];
        }
        const size_t numberOfBlocks = std::accumulate(blocksPerAxis.begin(),
                                                      blocksPerAxis.end(), 1,
                                                      std::multiplies<size_t>());
        const Coordinate3D blockStrides = {blocksPerAxis[1] * blocksPerAxis[2],
                                         blocksPerAxis[2], 1};

        //
        // compute the histograms
        //
        HistogramStorage histograms(numberOfBlocks);
        std::vector<size_t> blockSizes(numberOfBlocks, 1);
        for(int64_t i = 0; i < blocksPerAxis[0]; ++i) {
            for(int64_t j = 0; j < blocksPerAxis[1]; ++j) {
                for(int64_t k = 0; k < blocksPerAxis[2]; ++k) {
                    const size_t blockId = getBlockIndex(i, j, k, blockStrides);
                    Coordinate3D blockBegin = {i * blockShape[0],
                                             j * blockShape[1],
                                             k * blockShape[2]};
                    Coordinate3D blockEnd = {std::min((i + 1) * blockShape[0], shape[0]),
                                           std::min((j + 1) * blockShape[1], shape[1]),
                                           std::min((k + 1) * blockShape[2], shape[2])};

                    size_t & blockSize = blockSizes[blockId];
                    if(haveIgnoreLabel){
                        blockSize = computeHistogramWithIgnoreLabel(labels, blockBegin, blockEnd,
                                                                    histograms[blockId], ignoreLabel);
                    }
                    else {
                        blockSize = computeHistogram(labels, blockBegin, blockEnd, histograms[blockId]);
                    }
                }
            }
        }

        //
        // compute the affinties
        //
        for(int64_t i = 0; i < blocksPerAxis[0]; ++i) {
            for(int64_t j = 0; j < blocksPerAxis[1]; ++j) {
                for(int64_t k = 0; k < blocksPerAxis[2]; ++k) {

                    const size_t blockId = getBlockIndex(i, j, k, blockStrides);
                    const auto & histo = histograms[blockId];
                    const size_t blockSize = blockSizes[blockId];

                    if(i > 0) {
                        const size_t ngbId = getBlockIndex(i - 1, j, k, blockStrides);
                        const auto & ngbHisto = histograms[ngbId];
                        const double norm = blockSize * blockSizes[ngbId];
                        if(norm > 0) {
                            affs(0, i, j, k) = computeSingleAffinity(histo, ngbHisto, norm);
                            mask(0, i, j, k) = 1;
                        } else {
                            mask(0, i, j, k) = 0;
                        }
                    } else {
                        mask(0, i, j, k) = 0;
                    }

                    if(j > 0) {
                        const size_t ngbId = getBlockIndex(i, j - 1, k, blockStrides);
                        const auto & ngbHisto = histograms[ngbId];
                        const double norm = blockSize * blockSizes[ngbId];
                        if(norm > 0) {
                            affs(1, i, j, k) = computeSingleAffinity(histo, ngbHisto, norm);
                            mask(1, i, j, k) = 1;
                        } else {
                            mask(1, i, j, k) = 0;
                        }
                    } else {
                        mask(1, i, j, k) = 0;
                    }

                    if(k > 0) {
                        const size_t ngbId = getBlockIndex(i, j, k - 1, blockStrides);
                        const auto & ngbHisto = histograms[ngbId];
                        const double norm =  blockSize * blockSizes[ngbId];
                        if(norm > 0) {
                            affs(2, i, j, k) = computeSingleAffinity(histo, ngbHisto, norm);
                            mask(2, i, j, k) = 1;
                        } else {
                            mask(2, i, j, k) = 0;
                        }
                    } else {
                        mask(2, i, j, k) = 0;
                    }

                }
            }
        }
    }

}
