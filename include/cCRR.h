#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pybind11/pybind11.h> // must be first
#include <pybind11/numpy.h> // must be first

namespace py = pybind11;

int sigma(py::array_t<float> image_in, int nx, int ny, int sp);
