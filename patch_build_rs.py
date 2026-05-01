with open("rust/build.rs", "r") as f:
    content = f.read()

content = content.replace('.file("src/rnbo_bridge.cpp")', '.file("src/rnbo_bridge.cpp")\n        .file("src/tlsf.c")')
content = content.replace('println!("cargo:rerun-if-changed=src/rnbo_bridge.cpp");', 'println!("cargo:rerun-if-changed=src/rnbo_bridge.cpp");\n    println!("cargo:rerun-if-changed=src/tlsf.c");\n    println!("cargo:rerun-if-changed=src/tlsf.h");')

with open("rust/build.rs", "w") as f:
    f.write(content)
