fn main() {
    cxx_build::bridge("src/rnbo_module.rs")
        .file("src/rnbo_bridge.cpp")
        .file("src/tlsf.c")
        .include("src")
        .include("../RNBO_Integration/common")
        .flag_if_supported("-std=c++14")
        .compile("signals_core_rnbo");

    println!("cargo:rerun-if-changed=src/rnbo_bridge.cpp");
    println!("cargo:rerun-if-changed=src/tlsf.c");
    println!("cargo:rerun-if-changed=src/tlsf.h");
    println!("cargo:rerun-if-changed=src/rnbo_bridge.h");
    println!("cargo:rerun-if-changed=src/rnbo_module.rs");
}
