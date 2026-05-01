# RNBO Minimal Export Integration

## Overview
Phase 2 of the RNBO integration migrated away from a full C++ engine wrapper and instead adopts the **Minimal Export** environment as demonstrated in Cycling '74's `rnbo.example.baremetal` repository.

## Implementation Details
1. **Memory Allocator:** The C++ bridge uses the `tlsf` (Two-Level Segregated Fit) memory allocator to manage memory dynamically during initialization, mapping it to RNBO's required platform overrides for `malloc`, `free`, `realloc`, and `calloc`.
2. **Minimal Engine:** A minimal engine (`MyEngine`) inherited from `RNBO::MinimalEngine<>` is used in order to provide the bare-minimum event target interface expected by the RNBO runtime.
3. **Rust Bridge Responsibilities:**
    - The `cxx` bridge interfaces solely with `RnboHost`, an opaque C++ wrapper holding the generated `rnbomatic` instance and its memory pool.
    - Rust handles all block-size conversions (`flattened_inputs` and `flattened_outputs`) ensuring that data conforms to C-like contiguous slices across the FFI boundary.
    - The Rust `ModuleGraph` manages graph execution, keeping the C++ boundary fully stateless concerning topological sorting or overall block sequencing.

## Current Status
- Instance generation, parameter updates, `prepareToProcess`, and `process` block routines have been bridged successfully.
- Tests (such as `test_rnbo_module_compiled`) run smoothly with fixed block sizes (e.g., 64) without producing segmentation faults.
