#pragma once
#include <unordered_map>
#include <vector>
#include "xtensor/xtensor.hpp"


namespace multiscale_affinities {

    // typedefs
    typedef std::array<size_t, 3> Coordinate;
    typedef std::unordered_map<uint64_t, size_t> Histogram;
    typedef std::vector<Histogram> HistogramStorage;


    template<class LABELS>
    inline size_t computeHistogram(const LABELS & labels,
                                   const Coordinate & blockBegin,
                                   const Coordinate & blockEnd,
                                   Histogram & out) {
        size_t nPixels = 0;
        for(size_t z = blockBegin[0]; z < blockEnd[0]; ++z) {
            for(size_t y = blockBegin[1]; y < blockEnd[1]; ++y) {
                for(size_t x = blockBegin[2]; x < blockEnd[2]; ++x) {
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
    inline size_t computeHistogramWithIgnoreLabel(const LABELS & labels,
                                                  const Coordinate & blockBegin,
                                                  const Coordinate & blockEnd,
                                                  Histogram & out,
                                                  const uint64_t ignoreLabel) {
        size_t nPixels = 0;
        for(size_t z = blockBegin[0]; z < blockEnd[0]; ++z) {
            for(size_t y = blockBegin[1]; y < blockEnd[1]; ++y) {
                for(size_t x = blockBegin[2]; x < blockEnd[2]; ++x) {
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
                                const Coordinate & blockStrides) {
        return blockStrides[0] * i + blockStrides[1] * j + blockStrides[2] * k;
    }


    // TODO mask for valid / invalid affinities
    template<class LABEL_ARRAY, class AFFS_ARRAY>
    void computeMultiscaleAffinities(const xt::xexpression<LABEL_ARRAY> & labelsExp,
                                     const std::vector<int> & blockShape,
                                     xt::xexpression<AFFS_ARRAY> & affsExp,
                                     const bool haveIgnoreLabel=false,
                                     const uint64_t ignoreLabel=0) {

        const auto & labels = labelsExp.derived_cast();
        auto & affs = affsExp.derived_cast();

        //
        // compute the block sizes and number of blocks
        //
        Coordinate shape;
        Coordinate blocksPerAxis;
        for(unsigned d = 0; d < 3; ++d) {
            blocksPerAxis[d] = affs.shape()[d + 1];
            shape[d] = labels.shape()[d];
        }
        const size_t numberOfBlocks = std::accumulate(blocksPerAxis.begin(),
                                                      blocksPerAxis.end(), 1,
                                                      std::multiplies<size_t>());
        const Coordinate blockStrides = {blocksPerAxis[1] * blocksPerAxis[2],
                                         blocksPerAxis[2], 1};

        //
        // compute the histograms
        //
        HistogramStorage histograms(numberOfBlocks);
        std::vector<size_t> blockSizes(numberOfBlocks, 1);
        for(size_t i = 0; i < blocksPerAxis[0]; ++i) {
            for(size_t j = 0; j < blocksPerAxis[1]; ++j) {
                for(size_t k = 0; k < blocksPerAxis[2]; ++k) {
                    const size_t blockId = getBlockIndex(i, j, k, blockStrides);
                    Coordinate blockBegin = {i * blockShape[0],
                                             j * blockShape[1],
                                             k * blockShape[2]};
                    Coordinate blockEnd = {std::min((i + 1) * blockShape[0], shape[0]),
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
        for(size_t i = 0; i < blocksPerAxis[0]; ++i) {
            for(size_t j = 0; j < blocksPerAxis[1]; ++j) {
                for(size_t k = 0; k < blocksPerAxis[2]; ++k) {

                    const size_t blockId = getBlockIndex(i, j, k, blockStrides);
                    const auto & histo = histograms[blockId];
                    const size_t blockSize = blockSizes[blockId];

                    if(i > 0) {
                        const size_t ngbId = getBlockIndex(i - 1, j, k, blockStrides);
                        const auto & ngbHisto = histograms[ngbId];
                        const double norm = blockSize * blockSizes[ngbId];
                        affs(0, i, j, k) = computeSingleAffinity(histo, ngbHisto, norm);
                    }

                    if(j > 0) {
                        const size_t ngbId = getBlockIndex(i, j - 1, k, blockStrides);
                        const auto & ngbHisto = histograms[ngbId];
                        const double norm = blockSize * blockSizes[ngbId];
                        affs(1, i, j, k) = computeSingleAffinity(histo, ngbHisto, norm);
                    }

                    if(k > 0) {
                        const size_t ngbId = getBlockIndex(i, j, k - 1, blockStrides);
                        const auto & ngbHisto = histograms[ngbId];
                        const double norm =  blockSize * blockSizes[ngbId];
                        affs(2, i, j, k) = computeSingleAffinity(histo, ngbHisto, norm);
                    }

                }
            }
        }
    }

}
