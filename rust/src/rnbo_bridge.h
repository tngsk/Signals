#pragma once

#include "../../RNBO_Integration/common/RNBO_PatcherInterface.h"
#include <memory>
#include <vector>
#include "rust/cxx.h"

namespace rnbo_bridge {

class RnboHost {
public:
    RnboHost(double sample_rate, size_t block_size);
    ~RnboHost();

    void prepare_to_process(double sample_rate, size_t block_size);
    void set_parameter(size_t index, double value);
    double get_parameter(size_t index) const;

    void process_block(const rust::Slice<const double> inputs, rust::Slice<double> outputs, size_t block_size);

    size_t get_num_inputs() const;
    size_t get_num_outputs() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

std::unique_ptr<RnboHost> create_rnbo_host(double sample_rate, size_t block_size);

} // namespace rnbo_bridge
