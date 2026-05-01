#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "analogosc.cpp.h"

namespace py = pybind11;

class RNBOOscillatorWrapper {
public:
    RNBOOscillatorWrapper() {
        patcher.initialize();
        patcher.prepareToProcess(48000, 64, true);
    }

    void set_sample_rate(double sample_rate) {
        patcher.prepareToProcess(sample_rate, 64, true);
    }

    void set_parameter(int index, double value) {
        patcher.setParameterValueNormalized(index, value);
    }

    py::array_t<double> process(py::array_t<double> frequency) {
        py::buffer_info freq_buf = frequency.request();
        int block_size = freq_buf.shape[0];

        // Create an output array
        auto result = py::array_t<double>(block_size);
        py::buffer_info result_buf = result.request();
        double* result_ptr = static_cast<double*>(result_buf.ptr);

        double* in_ptr[1] = {static_cast<double*>(freq_buf.ptr)};

        // We have 2 outputs according to maxpat
        std::vector<double> out1(block_size, 0.0);
        std::vector<double> out2(block_size, 0.0);
        double* out_ptr[2] = {out1.data(), out2.data()};

        patcher.process(in_ptr, 1, out_ptr, 2, block_size);

        // Copy first output channel to numpy array
        for(int i = 0; i < block_size; i++) {
            result_ptr[i] = out1[i];
        }

        return result;
    }

private:
    RNBO::rnbomatic<> patcher;
};

PYBIND11_MODULE(rnbo_osc, m) {
    py::class_<RNBOOscillatorWrapper>(m, "RNBOOscillator")
        .def(py::init<>())
        .def("set_sample_rate", &RNBOOscillatorWrapper::set_sample_rate)
        .def("set_parameter", &RNBOOscillatorWrapper::set_parameter)
        .def("process", &RNBOOscillatorWrapper::process);
}
