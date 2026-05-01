with open("rust/tests/rnbo_tests.rs", "r") as f:
    content = f.read()

content = content.replace('let _ = RNBOModule::new;', 'let mut module = RNBOModule::with_config(44100.0, 64);\n    let inputs = vec![vec![0.0; 64]; 2];\n    let outputs = module.process(&inputs);\n    assert_eq!(outputs.len(), module.output_count());')

with open("rust/tests/rnbo_tests.rs", "w") as f:
    f.write(content)
