#include "xtensor/xtensor.hpp"
#include "affinities/util.hxx"


namespace affinities {

    // run bfs connecting all nodes that are linked
    // by edges with affinities larger than threshold
    // we assume that affinities have the offsets:
    // (2d)
    // [-1, 0]
    // [0, -1]
    // (3d)
    // [-1, 0, 0]
    // [0, -1, 0]
    // [0, 0, -1]
    template<class AFFS, class LABELS>
    inline void bfs(const xt::xexpression<AFFS> & affinitiesExp,
                    xt::xexpression<LABELS> & labelsExp,
                    const xt::xindex & coord,
                    const typename LABELS::value_type current_label,
                    const float threshold) {
        const auto & affs = affinitiesExp.derived_cast();
        auto & labels = labelsExp.derived_cast();
        // label the current pixel
        labels[coord] = current_label;

        const unsigned dim = labels.size();
        // initialise the affinity coordinate at the current pixel position
        xt::xindex affCoord(affs.dimension());
        std::copy(coord.begin(), coord.end(), affCoord.begin() + 1);
        // iterate over the adjacent pixels
        for(unsigned d = 0; d < dim; ++d) {
            // get the affinity of the edge to
            // this adjacent pixel
            affCoord[0] = d;
            const auto aff = affs[affCoord];
            // check whether the pixels are connected
            // according to the threshold TODO < or >
            if(aff < threshold) {
                continue;  // continue if not connected
            }
            // get the coordinate of adjacent pixel
            xt::xindex nextCoord = coord;
            --nextCoord[d];
            // check if the pixel is out of range
            if(nextCoord[d] < 0) {
                continue;  // continue if out of range
            }
            // continue bfs from adjacent node
            bfs(affs, labels, nextCoord, current_label, threshold);
        }
    }


    // compute connected components based on affinities
    template<class AFFS, class LABELS>
    inline size_t connected_components(const xt::xexpression<AFFS> & affinitiesExp,
                                       xt::xexpression<LABELS> & labelsExp,
                                       const float threshold){
        //
        typedef typename LABELS::value_type LabelType;
        const auto affs = affinitiesExp.derived_cast();
        auto & labels = labelsExp.derived_cast();

        //
        LabelType currentLabel = 1;
        xt::xindex shape(labels.shape().begin(), labels.shape().end());
        // iterate over the nodes (pixels), run bfs for each node
        // to label all connected nodes
        util::forEachCoordinate(shape, [&](const xt::xindex & coord){
            // don't do anything if this label is already labeled
            if(labels[coord] != 0) {
                return;
            }

            // run bfs beginning from the current node (pixel)
            bfs(affs, labels, coord, currentLabel, threshold);
            // increase the label
            ++currentLabel;
        });

        // return the max label
        return static_cast<size_t>(currentLabel - 1);
    }

}
