#pragma once

#include <algorithm>
#include <unordered_map>

#include "boost/pending/disjoint_sets.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xstrided_view.hpp"


namespace affinities {


    template<class GT, class OVERLAPS, class SEGMENT_SIZES, class UFD>
    inline void initMalis2d(const xt::xexpression<GT> & gtExp,
                            OVERLAPS & overlaps,
                            SEGMENT_SIZES & segmentSizes,
                            UFD & sets,
                            size_t & nNodesLabeled,
                            size_t & nPairPos) {
        const auto & gt = gtExp.derived_cast();
        size_t nodeIndex = 0;

        const auto & shape = gt.shape();
        for(int64_t i = 0; i < shape[0]; ++i) {
            for(int64_t j = 0; j < shape[1]; ++j) {
                const auto gtId = gt(i, j);
                if(gtId != 0) {
                    overlaps[nodeIndex].insert(std::make_pair(gtId, 1));
                    ++segmentSizes[gtId];
                    ++nNodesLabeled;
                    nPairPos += (segmentSizes[gtId] - 1);
                }
                ++nodeIndex;
            }
        }
    }


    template<class GT, class OVERLAPS, class SEGMENT_SIZES, class UFD>
    inline void initMalis3d(const xt::xexpression<GT> & gtExp,
                            OVERLAPS & overlaps,
                            SEGMENT_SIZES & segmentSizes,
                            UFD & sets,
                            size_t & nNodesLabeled,
                            size_t & nPairPos) {
        const auto & gt = gtExp.derived_cast();
        size_t nodeIndex = 0;

        const auto & shape = gt.shape();
        for(int64_t i = 0; i < shape[0]; ++i) {
            for(int64_t j = 0; j < shape[1]; ++j) {
                for(int64_t k = 0; k < shape[2]; ++k) {
                    const auto gtId = gt(i, j, k);
                    if(gtId != 0) {
                        overlaps[nodeIndex].insert(std::make_pair(gtId, 1));
                        ++segmentSizes[gtId];
                        ++nNodesLabeled;
                        nPairPos += (segmentSizes[gtId] - 1);
                    }
                    ++nodeIndex;
                }
            }
        }
    }

    template<class AFFS, class GT, class GRADS>
    double malis_gradient(const xt::xexpression<AFFS> & affinitiesExp,
                          const xt::xexpression<GT> & gtExp,
                          xt::xexpression<GRADS> & gradientsExp,
                          const std::vector<std::vector<int>> & offsets,
                          const bool pos) {
        const auto & affs = affinitiesExp.derived_cast();
        const auto & gt = gtExp.derived_cast();
        auto & grads = gradientsExp.derived_cast();

        typedef typename GT::value_type LabelType;
        typedef typename AFFS::value_type AffType;
        typedef typename GRADS::value_type GradType;
        // TODO check shapes
        const auto & shape = gt.shape();
        const auto & gtStrides = gt.strides();
        const unsigned nDim = gt.dimension();

        // nodes and edges in the affinity graph
        const size_t nNodes = gt.size();
        const size_t nEdges = affs.size();

        // make union find for the mst
        std::vector<LabelType> rank(nNodes);
        std::vector<LabelType> parent(nNodes);
        boost::disjoint_sets<LabelType*, LabelType*> sets(&rank[0], &parent[0]);

        // data structures for overlaps of nodes (= pixels) with gt labels
        // and sizes of gt segments
        std::vector<std::unordered_map<LabelType, size_t>> overlaps(nNodes);
        std::unordered_map<LabelType, size_t> segmentSizes;

        // initialize sets, overlaps and find labeled pixels
        size_t nNodesLabeled = 0, nPairPos = 0;
        if(nDim == 2) {
            initMalis2d(gt, overlaps, segmentSizes, sets, nNodesLabeled, nPairPos);
        } else {
            initMalis3d(gt, overlaps, segmentSizes, sets, nNodesLabeled, nPairPos);
        }

        // compute normalisation
        const size_t nPairNorm = pos ? nPairPos : nNodesLabeled * (nNodesLabeled - 1) / 2 - nPairPos;
        if(nPairNorm == 0) {
            throw std::runtime_error("Normalization is zero!");
        }

        // sort the edges by affinity strength
        const auto flatView = xt::flatten(affs); // flat view
        std::vector<size_t> pqueue(nEdges);
        std::iota(pqueue.begin(), pqueue.end(), 0);

        // sort in increasing order
        std::sort(pqueue.begin(), pqueue.end(), [&flatView](const size_t ind1,
                                                            const size_t ind2){
            return flatView(ind1) > flatView(ind2);
        });

        // run kruskal and calculate mals gradeint for each edge in the spanning tree
        // initialize values
        const auto & strides = affs.strides();
        const auto & layout = affs.layout();
        LabelType setU, setV, nodeU, nodeV;
        // coordinates for affinities and gt
        xt::xindex affCoord(affs.size());
        xt::xindex gtCoordU(gt.size()), gtCoordV(gt.size());

        double loss = 0;
        // iterate over the queue
        for(const auto edgeId : pqueue) {
            // translate edge id to coordinate
            const auto affCoord_ = xt::unravel_from_strides(edgeId, strides, layout);
            std::copy(affCoord_.begin(), affCoord_.end(), affCoord.begin());

            // get offset and spatial coordinates
            const auto & offset = offsets[affCoord[0]];
            // range check
            bool inRange = true;
            for(unsigned d = 0; d < nDim; ++d) {
                gtCoordU[d] = affCoord[d + 1];
                gtCoordV[d] = affCoord[d + 1] + offset[d];
                if(gtCoordV[d] < 0 || gtCoordV[d] >= shape[d]) {
                    inRange = false;
                    break;
                }
            }

            if(!inRange) {
                continue;
            }

            // get the spatial node index
            LabelType nodeU = 0, nodeV = 0;
            for(unsigned d = 0; d < nDim; ++d) {
                nodeU += gtCoordU[d] * gtStrides[d];
                nodeV += gtCoordV[d] * gtStrides[d];
            }

            // get the representatives of the nodes
            LabelType setU = sets.find_set(nodeU);
            LabelType setV = sets.find_set(nodeV);

            // check if the nodes are not merged yet
            if(setU != setV) {

                // merge nodes
                // sets.link(nodeU, nodeV);
                sets.link(setU, setV);

                // initialize values for gradient calculation
                GradType currentGradient = 0;
                const AffType aff = affs[affCoord];
                const GradType grad = pos ? 1. - aff : -aff;

                // compute the number of node pairs merged by this edge
                for(auto itU = overlaps[setU].begin(); itU != overlaps[setU].end(); ++itU) {
                    for(auto itV = overlaps[setV].begin(); itV != overlaps[setV].end(); ++itV) {
                        const size_t nPair = itU->second * itV->second;
                        if(pos && (itU->first == itV->first)) {
                            loss += grad * grad * nPair;
                            currentGradient += grad * nPair;
                        }

                        if(!pos && (itU->first != itV->first)) {
                            loss += grad * grad * nPair;
                            currentGradient += grad * nPair;
                        }
                    }
                }
                grads[affCoord] += currentGradient / nPairNorm;

                if(sets.find_set(setU) == setV) {
                    std::swap(setU, setV);
                }

                auto & overlapsU = overlaps[setU];
                auto & overlapsV = overlaps[setV];
                auto itV = overlapsV.begin();
                while(itV != overlapsV.end()) {
                    auto itU = overlapsU.find(itV->first);
                    if(itU == overlapsU.end()) {
                        overlapsU.insert(std::make_pair(itV->first, itV->second));
                    } else {
                        itU->second += itV->second;
                    }
                    overlapsV.erase(itV);
                    ++itV;
                }
            }
        }
        return loss / nPairNorm;
    }


}
