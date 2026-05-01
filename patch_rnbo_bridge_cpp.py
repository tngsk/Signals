import re

with open("rust/src/rnbo_bridge.cpp", "r") as f:
    content = f.read()

# Add a static reference counter
content = content.replace('tlsf_t myPool = nullptr;', 'tlsf_t myPool = nullptr;\nstatic int poolRefCount = 0;\nstatic void* memPoolBufferGlobal = nullptr;')

# Update calloc safely
calloc_block = """void* calloc(size_t count, size_t size) {
            auto mem = malloc(count * size);
            memset(mem, 0, count * size);
            return mem;
        }"""
safe_calloc_block = """void* calloc(size_t count, size_t size) {
            auto mem = malloc(count * size);
            if (mem) {
                memset(mem, 0, count * size);
            }
            return mem;
        }"""
content = content.replace(calloc_block, safe_calloc_block)

# Remove memPoolBuffer from Impl
content = re.sub(r'void\* memPoolBuffer;\s*Impl\(\) : rnbo\(nullptr\), memPoolBuffer\(nullptr\) \{\}', 'Impl() : rnbo(nullptr) {}', content)

# Fix constructor to handle ref counting
ctor_block = """RnboHost::RnboHost(double sample_rate, size_t block_size) : pImpl(std::make_unique<Impl>()) {
    if (!myPool) {
        const size_t poolSize = 10 * 1024 * 1024; // 10 MB should be more than enough for rnbo core objects
        pImpl->memPoolBuffer = ::malloc(poolSize);
        myPool = tlsf_create_with_pool(pImpl->memPoolBuffer, poolSize);
    }

    pImpl->rnbo = std::make_unique<RNBO::rnbomatic<MyEngine>>();
    pImpl->rnbo->initialize();
    pImpl->rnbo->prepareToProcess(sample_rate, block_size, true);
}"""
safe_ctor_block = """RnboHost::RnboHost(double sample_rate, size_t block_size) : pImpl(std::make_unique<Impl>()) {
    if (poolRefCount == 0) {
        const size_t poolSize = 10 * 1024 * 1024; // 10 MB should be more than enough for rnbo core objects
        memPoolBufferGlobal = ::malloc(poolSize);
        myPool = tlsf_create_with_pool(memPoolBufferGlobal, poolSize);
    }
    poolRefCount++;

    pImpl->rnbo = std::make_unique<RNBO::rnbomatic<MyEngine>>();
    pImpl->rnbo->initialize();
    pImpl->rnbo->prepareToProcess(sample_rate, block_size, true);
}"""
content = content.replace(ctor_block, safe_ctor_block)

# Fix destructor
dtor_block = """RnboHost::~RnboHost() {
    pImpl->rnbo.reset();
    if (myPool && pImpl->memPoolBuffer) {
        tlsf_destroy(myPool);
        ::free(pImpl->memPoolBuffer);
        myPool = nullptr;
    }
}"""
safe_dtor_block = """RnboHost::~RnboHost() {
    pImpl->rnbo.reset();
    poolRefCount--;
    if (poolRefCount == 0 && myPool) {
        tlsf_destroy(myPool);
        ::free(memPoolBufferGlobal);
        myPool = nullptr;
        memPoolBufferGlobal = nullptr;
    }
}"""
content = content.replace(dtor_block, safe_dtor_block)

# Add prepare_to_process to class
content = content.replace('void RnboHost::set_parameter(size_t index, double value) {', 'void RnboHost::prepare_to_process(double sample_rate, size_t block_size) {\n    if (pImpl->rnbo) pImpl->rnbo->prepareToProcess(sample_rate, block_size, true);\n}\n\nvoid RnboHost::set_parameter(size_t index, double value) {')

with open("rust/src/rnbo_bridge.cpp", "w") as f:
    f.write(content)
