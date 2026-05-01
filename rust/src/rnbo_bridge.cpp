#include "rnbo_bridge.h"
#include "../../RNBO_Integration/analogosc.cpp.h"

class StdOutLogger : public RNBO::Logger {
public:
    void log(RNBO::LogLevel level, const char* message) override {}
};

static StdOutLogger stdOutLogger;

namespace rnbo_bridge {

RNBOObject::RNBOObject() {
    RNBO::SetLogger(&stdOutLogger);
    rnbo_object.reset(RNBO::GetPatcherFactoryFunction()());
    if (rnbo_object) {
        rnbo_object->prepareToProcess(44100, 256);
    }
}

RNBOObject::~RNBOObject() = default;

void RNBOObject::set_parameter(size_t index, double value) {
    if (rnbo_object) rnbo_object->setParameterValue(index, value);
}

double RNBOObject::get_parameter(size_t index) const {
    if (rnbo_object) return rnbo_object->getParameterValue(index);
    return 0.0;
}

void RNBOObject::process(const rust::Slice<const double> input, rust::Slice<double> output) {
    if (!rnbo_object) return;
    const double* inputs[] = { input.data() };
    double* outputs[] = { output.data() };
    rnbo_object->process(inputs, 1, outputs, 1, input.size());
}

void RNBOObject::process_block(const rust::Slice<const double> inputs, rust::Slice<double> outputs, size_t block_size) {
    if (!rnbo_object) return;

    size_t num_inputs = rnbo_object->getNumInputChannels();
    size_t num_outputs = rnbo_object->getNumOutputChannels();

    std::vector<const double*> in_ptrs(num_inputs, nullptr);
    std::vector<double*> out_ptrs(num_outputs, nullptr);

    std::vector<double> dummy_in(block_size, 0.0);
    std::vector<double> dummy_out(block_size, 0.0);

    for (size_t i = 0; i < num_inputs; ++i) {
        if ((i + 1) * block_size <= inputs.size()) {
            in_ptrs[i] = inputs.data() + (i * block_size);
        } else {
            in_ptrs[i] = dummy_in.data();
        }
    }

    for (size_t i = 0; i < num_outputs; ++i) {
        if ((i + 1) * block_size <= outputs.size()) {
            out_ptrs[i] = outputs.data() + (i * block_size);
        } else {
            out_ptrs[i] = dummy_out.data();
        }
    }

    rnbo_object->process(in_ptrs.data(), num_inputs, out_ptrs.data(), num_outputs, block_size);
}

size_t RNBOObject::get_num_inputs() const {
    if (rnbo_object) return rnbo_object->getNumInputChannels();
    return 0;
}

size_t RNBOObject::get_num_outputs() const {
    if (rnbo_object) return rnbo_object->getNumOutputChannels();
    return 0;
}

std::unique_ptr<RNBOObject> create_rnbo_object() {
    return std::make_unique<RNBOObject>();
}

} // namespace rnbo_bridge
