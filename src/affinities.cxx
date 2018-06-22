#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyarray.hpp"

#include "affinities/affinities.hxx"
#include "affinities/connected_components.hxx"
#include "affinities/malis.hxx"
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


    // TODO use constrained malis once implemented
    m.def("compute_malis_2d", [](const xt::pytensor<float, 3> & affinities,
                                 const xt::pytensor<uint64_t, 2> & labels,
                                 const std::vector<std::vector<int>> & offsets) {
        //
        const auto & affShape = affinities.shape();
        double loss;
        xt::pytensor<float, 3> gradients(affShape);
        {
            py::gil_scoped_release allowThreads;
            loss = affinities::malis_gradient(affinities, labels,
                                              gradients, offsets, true);
        }
        return std::make_pair(loss, gradients);
    }, py::arg("affinities"),
       py::arg("labels"),
       py::arg("offsets"));


    m.def("connected_components", [](const xt::pyarray<float> & affinities,
                                     const float threshold) {
        typedef xt::pyarray<uint64_t>::shape_type ShapeType;
        ShapeType shape(affinities.shape().begin() + 1, affinities.shape().end());
        xt::pyarray<uint64_t> labels = xt::zeros<uint64_t>(shape);
        size_t max_label;
        {
            py::gil_scoped_release allowThreads;
            max_label = affinities::connected_components(affinities, labels, threshold);
        }
        return std::make_pair(labels, max_label);
    }, py::arg("affinities"),
       py::arg("threshold")
    );

}
