#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "cCRR.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(cCRR, m) {

	m.def("sigma", &sigma);
}
