#pragma once
#include <unordered_map>
#include <vector>
#include "xtensor/xtensor.hpp"


namespace affinities {

    typedef std::array<int64_t, 3> Coordinate;

    template<class LABEL_ARRAY, class AFFS_ARRAY, class MASK_ARRAY>
    void computeAffinities(const xt::xexpression<LABEL_ARRAY> & labelsExp,
                           const std::vector<std::array<int, 3>> & offsets,
                           xt::xexpression<AFFS_ARRAY> & affsExp,
                           xt::xexpression<MASK_ARRAY> & maskExp,
                           const bool haveIgnoreLabel=false,
                           const uint64_t ignoreLabel=0) {
        const auto & labels = labelsExp.derived_cast();
        auto & affs = affsExp.derived_cast();
        auto & mask = maskExp.derived_cast();

        typedef typename AFFS_ARRAY::value_type AffinityType;

        const Coordinate shape = {labels.shape()[0],
                                  labels.shape()[1],
                                  labels.shape()[2]};
        const int64_t nChannels = offsets.size();

        // TODO I don't know if it matters terribly much, but in terms of
        // cache locality, this iteration order seems to be not really optimal,
        // because we iterate over the complete label image `nChannels` times
        for(int64_t c = 0; c < nChannels; ++c) {
            for(int64_t i = 0; i < shape[0]; ++i) {
                for(int64_t j = 0; j < shape[1]; ++j) {
                    for(int64_t k = 0; k < shape[2]; ++k) {

                        Coordinate ngb = {i, j, k};
                        const auto & offset = offsets[c];

                        bool outOfRange = false;
                        for(unsigned d = 0; d < 3; ++d) {
                            ngb[d] += offset[d];
                            if(ngb[d] < 0 || ngb[d] >= shape[d]) {
                                outOfRange = true;
                                mask(c, i, j, k) = 0;
                                break;
                            }
                        }

                        if(outOfRange) {
                            continue;
                        }

                        const uint64_t label = labels(i, j, k);
                        const uint64_t labelNgb = labels(ngb[0], ngb[1], ngb[2]);

                        if(haveIgnoreLabel) {
                            if(label == ignoreLabel || labelNgb == ignoreLabel) {
                                mask(c, i, j, k) = 0;
                                continue;
                            }
                        }

                        affs(c, i, j, k) = static_cast<AffinityType>(label == labelNgb);
                        mask(c, i, j, k) = 1;
                    }
                }
            }
        }
    }

}
