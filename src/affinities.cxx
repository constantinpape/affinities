#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"

#include "affinities/affinities.hxx"
#include "affinities/multiscale_affinities.hxx"
#include "affinities/fullscale_multiscale_affinities.hxx"

namespace py = pybind11;

PYBIND11_MODULE(affinities, m)
{
    xt::import_numpy();

    m.doc() = R"pbdoc(
        Fast computation of affinities and multi-scale affinities

        .. currentmodule:: affinities

        .. autosummary::
           :toctree: _generate

           compute_affinities
           compute_multiscale_affinities
    )pbdoc";


    m.def("compute_multiscale_affinities_2d", [](const xt::pytensor<uint64_t, 2> & labels,
                                                 const std::vector<int> & blockShape,
                                                 const bool haveIgnoreLabel,
                                                 const uint64_t ignoreLabel) {
            // compute the out shape
            typedef typename xt::pytensor<float, 3>::shape_type ShapeType;
            const auto & shape = labels.shape();
            ShapeType outShape;
            outShape[0] = 2;
            for(unsigned d = 0; d < 2; ++d) {
                // integer division should do the right thing in all cases
                outShape[d + 1] = (shape[d] % blockShape[d]) ? shape[d] / blockShape[d] + 1 : shape[d] / blockShape[d];
            }

            // allocate the output
            xt::pytensor<float, 3> affs(outShape);
            xt::pytensor<uint8_t, 3> mask(outShape);
            {
                py::gil_scoped_release allowThreads;
                affinities::computeMultiscaleAffinities2D(labels, blockShape,
                                                          affs, mask,
                                                          haveIgnoreLabel, ignoreLabel);
            }
            return std::make_pair(affs, mask);
        }, py::arg("labels"),
           py::arg("blockShape"),
           py::arg("haveIgnoreLabel")=false,
           py::arg("ignoreLabel")=0);


    m.def("compute_multiscale_affinities_3d", [](const xt::pytensor<uint64_t, 3> & labels,
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
                outShape[d + 1] = (shape[d] % blockShape[d]) ? shape[d] / blockShape[d] + 1 : shape[d] / blockShape[d];
            }

            // allocate the output
            xt::pytensor<float, 4> affs(outShape);
            xt::pytensor<uint8_t, 4> mask(outShape);
            {
                py::gil_scoped_release allowThreads;
                affinities::computeMultiscaleAffinities3D(labels, blockShape,
                                                          affs, mask,
                                                          haveIgnoreLabel, ignoreLabel);
            }
            return std::make_pair(affs, mask);
        }, py::arg("labels"),
           py::arg("blockShape"),
           py::arg("haveIgnoreLabel")=false,
           py::arg("ignoreLabel")=0);


    m.def("compute_fullscale_multiscale_affinities", [](const xt::pytensor<uint64_t, 3> & labels,
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
                outShape[d + 1] = shape[d];
            }

            // allocate the output
            xt::pytensor<float, 4> affs(outShape);
            xt::pytensor<uint8_t, 4> mask(outShape);
            {
                py::gil_scoped_release allowThreads;
                affinities::computeFullscaleMultiscaleAffinities(labels, blockShape,
                                                                 affs, mask,
                                                                 haveIgnoreLabel, ignoreLabel);
            }
            return std::make_pair(affs, mask);
        }, py::arg("labels"),
           py::arg("blockShape"),
           py::arg("haveIgnoreLabel")=false,
           py::arg("ignoreLabel")=0);


    m.def("compute_affinities_2d", [](const xt::pytensor<uint64_t, 2> & labels,
                                      const std::vector<std::array<int, 2>> & offsets,
                                      const bool haveIgnoreLabel,
                                      const uint64_t ignoreLabel) {
            // compute the out shape
            typedef typename xt::pytensor<float, 3>::shape_type ShapeType;
            const auto & shape = labels.shape();
            ShapeType outShape;
            outShape[0] = offsets.size();
            for(unsigned d = 0; d < 2; ++d) {
                outShape[d + 1] = shape[d];
            }

            // allocate the output
            xt::pytensor<float, 3> affs(outShape);
            xt::pytensor<uint8_t, 3> mask(outShape);
            {
                py::gil_scoped_release allowThreads;
                affinities::computeAffinities2D(labels, offsets,
                                                affs, mask,
                                                haveIgnoreLabel, ignoreLabel);
            }
            return std::make_pair(affs, mask);
        }, py::arg("labels"),
           py::arg("offset"),
           py::arg("haveIgnoreLabel")=false,
           py::arg("ignoreLabel")=0);


    m.def("compute_affinities_3d", [](const xt::pytensor<uint64_t, 3> & labels,
                                      const std::vector<std::array<int, 3>> & offsets,
                                      const bool haveIgnoreLabel,
                                      const uint64_t ignoreLabel) {
            // compute the out shape
            typedef typename xt::pytensor<float, 4>::shape_type ShapeType;
            const auto & shape = labels.shape();
            ShapeType outShape;
            outShape[0] = offsets.size();
            for(unsigned d = 0; d < 3; ++d) {
                outShape[d + 1] = shape[d];
            }

            // allocate the output
            xt::pytensor<float, 4> affs(outShape);
            xt::pytensor<uint8_t, 4> mask(outShape);
            {
                py::gil_scoped_release allowThreads;
                affinities::computeAffinities3D(labels, offsets,
                                                affs, mask,
                                                haveIgnoreLabel, ignoreLabel);
            }
            return std::make_pair(affs, mask);
        }, py::arg("labels"),
           py::arg("offset"),
           py::arg("haveIgnoreLabel")=false,
           py::arg("ignoreLabel")=0);
}
