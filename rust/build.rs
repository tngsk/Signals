fn main() {
    cxx_build::bridge("src/rnbo_module.rs")
        .file("src/rnbo_bridge.cpp")
        .include("src")
        .include("../RNBO_Integration/common")
        .flag_if_supported("-std=c++14")
        .compile("signals_core_rnbo");

    println!("cargo:rerun-if-changed=src/rnbo_bridge.cpp");
    println!("cargo:rerun-if-changed=src/rnbo_bridge.h");
    println!("cargo:rerun-if-changed=src/rnbo_module.rs");
}
