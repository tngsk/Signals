# RNBO C++ to Rust Execution Flow

This document provides a technical deep-dive into how RNBO C++ exported patches are loaded and executed natively within the Signals Rust Core Engine.

## Overview
The architecture is designed to completely decouple RNBO C++ code from complex framework dependencies (like JUCE or Python) by utilizing RNBO's "Minimal Export" target and binding it directly to Rust using `cxx`.

The processing chain looks like this:
`RNBO Minimal Export (C++)` -> `rnbo_bridge (C++)` -> `cxx (FFI)` -> `RNBOModule (Rust)`

## 1. The C++ Layer: Minimal Export & Custom Allocation

We use the RNBO Minimal Export which provides pure C++ DSP code without an underlying OS audio loop.

### Custom Memory Allocation (`tlsf`)
RNBO objects often require memory allocations during instantiation or when processing complex events. To ensure real-time safety and prevent segmentation faults during initialization without a host OS context, we override the platform memory allocators using **TLSF** (Two-Level Segregated Fit) memory allocator.

In `rnbo_bridge.cpp`:
```cpp
#define RNBO_USECUSTOMALLOCATOR
#include "tlsf.h"

// We create a global memory pool
tlsf_t myPool = nullptr;

namespace RNBO {
    namespace Platform {
        void* malloc(size_t size) { return tlsf_malloc(myPool, size); }
        void free(void* ptr) { tlsf_free(myPool, ptr); }
        // ...
    }
}
```

### The `RnboHost` Wrapper
The `RnboHost` C++ class wraps the `RNBO::MinimalEngine<>`. It handles the initialization of the RNBO patch (`rnbomatic<MyEngine>`), sets parameters, and exposes a `process_block` method that takes `rust::Slice` buffers.

```cpp
void RnboHost::process_block(const rust::Slice<const double> inputs, rust::Slice<double> outputs, size_t block_size) {
    // Marshals C++ raw pointers from the rust::Slice and calls the RNBO engine
    pImpl->rnbo->process(in_ptrs.data(), num_inputs, out_ptrs.data(), num_outputs, block_size);
}
```

## 2. The Rust Bridge Layer: `cxx`

The `cxx` crate securely bridges the C++ `RnboHost` object into Rust, guaranteeing memory safety across the FFI boundary.

In `rnbo_module.rs`, the `cxx::bridge` macro exposes the `RnboHost` C++ functions to Rust:

```rust
#[cxx::bridge(namespace = "rnbo_bridge")]
pub mod ffi {
    unsafe extern "C++" {
        include!("signals_core/src/rnbo_bridge.h");
        type RnboHost;
        fn create_rnbo_host(sample_rate: f64, block_size: usize) -> UniquePtr<RnboHost>;
        fn process_block(self: Pin<&mut RnboHost>, inputs: &[f64], outputs: &mut [f64], block_size: usize);
        // ...
    }
}
```

## 3. The Rust Execution Layer: `RNBOModule`

In the Rust core engine, modules conform to the `Module` trait, which expects a `process` method that takes and returns `Vec<Vec<f64>>` (blocks of audio data per channel).

The `RNBOModule` struct wraps the `cxx::UniquePtr<ffi::RnboHost>`.

When the graph scheduler calls `process` on the `RNBOModule`:
1. It inspects the size of the incoming buffers to determine `actual_block_size`.
2. It flattens the multi-channel 2D `Vec<Vec<f64>>` inputs into a 1D `Vec<f64>` array because C++ expects contiguous memory.
3. It passes these contiguous slices to `process_block` via `pin_mut()`.
4. It reconstructs the 1D C++ output array back into the 2D `Vec<Vec<f64>>` structure expected by the Rust `ModuleGraph`.

```rust
impl Module for RNBOModule {
    fn process(&mut self, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        // Flatten inputs...

        // Execute C++ block via cxx bridge
        self.inner.pin_mut().process_block(&flattened_inputs, &mut flattened_outputs, actual_block_size);

        // Unflatten outputs...
        return outputs;
    }
}
```

This ensures high-performance, block-based processing directly between the natively-compiled C++ DSP and the Rust routing graph, completely eliminating the overhead of legacy Python `pybind11` integrations.
