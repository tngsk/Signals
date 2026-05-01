#pragma once

#include "../../RNBO_Integration/common/RNBO_PatcherInterface.h"
#include <memory>
#include <vector>
#include "rust/cxx.h"

namespace rnbo_bridge {

class RNBOObject {
public:
    RNBOObject();
    ~RNBOObject();

    void set_parameter(size_t index, double value);
    double get_parameter(size_t index) const;

    void process(const rust::Slice<const double> input, rust::Slice<double> output);
    void process_block(const rust::Slice<const double> inputs, rust::Slice<double> outputs, size_t block_size);

    size_t get_num_inputs() const;
    size_t get_num_outputs() const;

private:
    std::unique_ptr<RNBO::PatcherInterface> rnbo_object;
};

std::unique_ptr<RNBOObject> create_rnbo_object();

} // namespace rnbo_bridge
