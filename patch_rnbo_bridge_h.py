with open("rust/src/rnbo_bridge.h", "r") as f:
    content = f.read()

content = content.replace('void set_parameter(size_t index, double value);', 'void prepare_to_process(double sample_rate, size_t block_size);\n    void set_parameter(size_t index, double value);')

with open("rust/src/rnbo_bridge.h", "w") as f:
    f.write(content)
