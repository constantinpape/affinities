#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"

#include "multiscale_affinities/multiscale_affinities.hxx"

namespace py = pybind11;

PYBIND11_MODULE(multiscale_affinities, m)
{
    xt::import_numpy();

    m.doc() = R"pbdoc(
        Fast computation of multi-scale affinities

        .. currentmodule:: multiscale_affinities

        .. autosummary::
           :toctree: _generate

           compute_multiscale_affinities
    )pbdoc";

    // TODO mask for valid / invalid affinities ?!
    m.def("compute_multiscale_affinities", [](const xt::pytensor<uint64_t, 3> & labels,
                                              const std::vector<int> & blockShape,
                                              const bool haveIgnoreLabel,
                                              const uint64_t ignoreLabel) {
            // compute the out shape
            typedef typename xt::pytensor<float, 4>::shape_type ShapeType;
            const auto & shape = labels.shape();
            ShapeType outShape;
            outShape[0] = 3;
            for(unsigned d = 0; d < 3; ++d) {
                // integer division should do the right thing in all cases
                outShape[d + 1] = (shape[d] % blockShape[d]) ? shape[d] / blockShape[d] + 1 : shape[d]/ blockShape[d];
            }

            // allocate the output
            xt::pytensor<float, 4> affs(outShape);
            {
                py::gil_scoped_release allowThreads;
                multiscale_affinities::computeMultiscaleAffinities(labels, blockShape, affs, haveIgnoreLabel, ignoreLabel);
            }
            return affs;
        },py::arg("labels"), py::arg("samplingFactors"), py::arg("haveIgnoreLabel")=false, py::arg("ignoreLabel")=0);
}
