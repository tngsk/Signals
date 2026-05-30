import os

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    content = content.replace("let inputs = vec![vec![0.0; 64]; 2];", """let buf1 = vec![0.0; 64];
    let buf2 = vec![0.0; 64];
    let inputs: Vec<&[f64]> = vec![&buf1, &buf2];""")
    with open(filepath, 'w') as f:
        f.write(content)

fix_file('rust/tests/rnbo_tests.rs')
