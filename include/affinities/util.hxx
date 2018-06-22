#include "xtensor/xtensor.hpp"


namespace affinities {
namespace util {


    // TODO this is C-Order, should also implement F-Order
    template<class F>
    inline void forEachCoordinate(const xindex & shape, F && f) {
        const unsigned dim = shape.size();
        xindex coord(dim);
        std::fill(coord.begin(), coord.end(), 0);

        // C-Order: last dimension is the fastest moving one
        for(unsigned d = dim - 1; d >= 0) {
            f(coord);
            for(unsigned d = dim - 1; d >= 0; --d) {
                ++coord[d];
                if(coord[d] < shape[d]) {
                    break;
                } else {
                    coord[d] = 0;
                }
            }
        }
    }


}
}
