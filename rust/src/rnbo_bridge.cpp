//
// rnbo_bridge.cpp
// Based on rnbo.example.baremetal (Cycling '74)
// Prerequisites: RNBO Minimal Export, fixed block size processing
//

#define RNBO_NOTHROW
#define RNBO_USECUSTOMPLATFORMPRINT
#define RNBO_USECUSTOMALLOCATOR
#define RNBO_FIXEDLISTSIZE 64

#include "tlsf.h"
#include <iostream>
#include <memory>
#include <vector>
#include <cstring>

tlsf_t myPool = nullptr;
static int poolRefCount = 0;
static void* memPoolBufferGlobal = nullptr;

namespace RNBO {
    namespace Platform {
        void* malloc(size_t size) {
            return tlsf_malloc(myPool, size);
        }
        void free(void* ptr) {
            tlsf_free(myPool, ptr);
        }
        void* realloc(void* ptr, size_t size) {
            return tlsf_realloc(myPool, ptr, size);
        }
        void* calloc(size_t count, size_t size) {
            auto mem = malloc(count * size);
            if (mem) {
                memset(mem, 0, count * size);
            }
            return mem;
        }

        static void printMessage(const char* message) {
            std::cout << message << std::endl;
        }
        static void printErrorMessage(const char* message) {
            printMessage(message);
        }
    }
}

#include "rnbo_bridge.h"
#include "../../RNBO_Integration/analogosc.cpp.h"

// MyEngine - minimal engine derived from MinimalEngine
class MyEngine : public RNBO::MinimalEngine<> {
public:
    MyEngine(RNBO::PatcherInterface* patcher)
        : RNBO::MinimalEngine<>(patcher) {}
};

namespace rnbo_bridge {

struct RnboHost::Impl {
    std::unique_ptr<RNBO::rnbomatic<MyEngine>> rnbo;
    Impl() : rnbo(nullptr) {}
};

RnboHost::RnboHost(double sample_rate, size_t block_size) : pImpl(std::make_unique<Impl>()) {
    if (poolRefCount == 0) {
        const size_t poolSize = 10 * 1024 * 1024; // 10 MB should be more than enough for rnbo core objects
        memPoolBufferGlobal = ::malloc(poolSize);
        myPool = tlsf_create_with_pool(memPoolBufferGlobal, poolSize);
    }
    poolRefCount++;

    pImpl->rnbo = std::make_unique<RNBO::rnbomatic<MyEngine>>();
    pImpl->rnbo->initialize();
    pImpl->rnbo->prepareToProcess(sample_rate, block_size, true);
}

RnboHost::~RnboHost() {
    pImpl->rnbo.reset();
    poolRefCount--;
    if (poolRefCount == 0 && myPool) {
        tlsf_destroy(myPool);
        ::free(memPoolBufferGlobal);
        myPool = nullptr;
        memPoolBufferGlobal = nullptr;
    }
}

void RnboHost::prepare_to_process(double sample_rate, size_t block_size) {
    if (pImpl->rnbo) pImpl->rnbo->prepareToProcess(sample_rate, block_size, true);
}

void RnboHost::set_parameter(size_t index, double value) {
    if (pImpl->rnbo) pImpl->rnbo->setParameterValue(index, value, RNBO::TimeNow);
}

double RnboHost::get_parameter(size_t index) const {
    if (pImpl->rnbo) return pImpl->rnbo->getParameterValue(index);
    return 0.0;
}

void RnboHost::process_block(const rust::Slice<const double> inputs, rust::Slice<double> outputs, size_t block_size) {
    if (!pImpl->rnbo) return;

    size_t num_inputs = pImpl->rnbo->getNumInputChannels();
    size_t num_outputs = pImpl->rnbo->getNumOutputChannels();

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

    pImpl->rnbo->process(in_ptrs.data(), num_inputs, out_ptrs.data(), num_outputs, block_size);
}

size_t RnboHost::get_num_inputs() const {
    if (pImpl->rnbo) return pImpl->rnbo->getNumInputChannels();
    return 0;
}

size_t RnboHost::get_num_outputs() const {
    if (pImpl->rnbo) return pImpl->rnbo->getNumOutputChannels();
    return 0;
}

std::unique_ptr<RnboHost> create_rnbo_host(double sample_rate, size_t block_size) {
    return std::make_unique<RnboHost>(sample_rate, block_size);
}

} // namespace rnbo_bridge
