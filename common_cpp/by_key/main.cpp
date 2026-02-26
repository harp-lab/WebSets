/**
 * Unified Set Operations By Key using Balanced Path + Decoupled Lookback (C++ / wgpu-native)
 *
 * Two-phase GPU pipeline:
 *   1. DPI (Diagonal Path Intersection) - compute merge path partition boundaries (keys only)
 *   2. Decoupled Lookback By Key - single-pass count + scan + write key-value pairs
 *
 * By-Key variant: compares by keys but preserves key-value pairs in output.
 *
 * Supports all 4 set operations via OP_MODE string replacement:
 *   0 = intersection (A ∩ B)
 *   1 = difference   (A \ B)
 *   2 = union        (A ∪ B)
 *   3 = sym_difference ((A\B) ∪ (B\A))
 *
 * Requires wgpu-native v24+ for subgroup support.
 */

#include <webgpu/webgpu.h>
#include <webgpu/wgpu.h>   // wgpu-native extras (wgpuDevicePoll, native features)

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Helper: WGPUStringView from C string
// ---------------------------------------------------------------------------
static WGPUStringView wgpuStr(const char* s) {
    return {s, s ? strlen(s) : 0};
}

// ---------------------------------------------------------------------------
// Constants (must match WGSL shaders)
// ---------------------------------------------------------------------------
static constexpr uint32_t NT = 256;     // threads per workgroup
static constexpr uint32_t VT = 12;      // values per thread
static constexpr uint32_t NV = NT * VT; // 3072 elements per workgroup
static constexpr uint32_t MAX_DISPATCH_X = 65535;
static constexpr uint32_t DPI_WORKGROUP_SIZE = 256;

static const char* OP_NAMES[] = {"intersection", "difference", "union", "sym_difference"};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string readFile(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        fprintf(stderr, "ERROR: cannot open file: %s\n", path);
        exit(1);
    }
    auto sz = f.tellg();
    f.seekg(0);
    std::string buf(static_cast<size_t>(sz), '\0');
    f.read(buf.data(), sz);
    return buf;
}

static std::vector<uint32_t> loadBinaryData(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        fprintf(stderr, "ERROR: cannot open binary file: %s\n", path);
        exit(1);
    }
    uint64_t count = 0;
    f.read(reinterpret_cast<char*>(&count), 8);
    std::vector<uint32_t> data(static_cast<size_t>(count));
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count * 4));
    return data;
}

// ---------------------------------------------------------------------------
// Generate synthetic values for by-key testing
// ---------------------------------------------------------------------------
static std::vector<uint32_t> generateValues(size_t count, uint32_t offset) {
    std::vector<uint32_t> values(count);
    for (size_t i = 0; i < count; i++) values[i] = (uint32_t)i + offset;
    return values;
}

// ---------------------------------------------------------------------------
// OP_MODE string replacement in shader source
// ---------------------------------------------------------------------------
static std::string setOpMode(const std::string& code, uint32_t opMode) {
    const std::string needle = "const OP_MODE: u32 = ";
    auto pos = code.find(needle);
    if (pos == std::string::npos) return code;
    auto semi = code.find(';', pos);
    if (semi == std::string::npos) return code;
    std::string result = code;
    result.replace(pos, semi - pos + 1,
        needle + std::to_string(opMode) + "u;");
    return result;
}

// ---------------------------------------------------------------------------
// CPU reference implementations (sorted multisets, by key)
// All operate on separate keys/values vectors, return pair of vectors.
// ---------------------------------------------------------------------------
using KVResult = std::pair<std::vector<uint32_t>, std::vector<uint32_t>>;

static KVResult cpuSetIntersectionByKey(const std::vector<uint32_t>& aKeys,
                                        const std::vector<uint32_t>& aVals,
                                        const std::vector<uint32_t>& bKeys,
                                        const std::vector<uint32_t>& bVals) {
    std::vector<uint32_t> rKeys, rVals;
    size_t ai = 0, bi = 0;
    while (ai < aKeys.size() && bi < bKeys.size()) {
        if (aKeys[ai] < bKeys[bi])      { ++ai; }
        else if (aKeys[ai] > bKeys[bi]) { ++bi; }
        else { rKeys.push_back(aKeys[ai]); rVals.push_back(aVals[ai]); ++ai; ++bi; }
    }
    return {rKeys, rVals};
}

static KVResult cpuSetDifferenceByKey(const std::vector<uint32_t>& aKeys,
                                      const std::vector<uint32_t>& aVals,
                                      const std::vector<uint32_t>& bKeys,
                                      const std::vector<uint32_t>& bVals) {
    (void)bVals;
    std::vector<uint32_t> rKeys, rVals;
    size_t ai = 0, bi = 0;
    while (ai < aKeys.size() && bi < bKeys.size()) {
        if (aKeys[ai] < bKeys[bi])      { rKeys.push_back(aKeys[ai]); rVals.push_back(aVals[ai]); ++ai; }
        else if (aKeys[ai] > bKeys[bi]) { ++bi; }
        else { ++ai; ++bi; }
    }
    while (ai < aKeys.size()) { rKeys.push_back(aKeys[ai]); rVals.push_back(aVals[ai]); ++ai; }
    return {rKeys, rVals};
}

static KVResult cpuSetUnionByKey(const std::vector<uint32_t>& aKeys,
                                 const std::vector<uint32_t>& aVals,
                                 const std::vector<uint32_t>& bKeys,
                                 const std::vector<uint32_t>& bVals) {
    std::vector<uint32_t> rKeys, rVals;
    size_t ai = 0, bi = 0;
    while (ai < aKeys.size() && bi < bKeys.size()) {
        if (aKeys[ai] < bKeys[bi])      { rKeys.push_back(aKeys[ai]); rVals.push_back(aVals[ai]); ++ai; }
        else if (aKeys[ai] > bKeys[bi]) { rKeys.push_back(bKeys[bi]); rVals.push_back(bVals[bi]); ++bi; }
        else { rKeys.push_back(aKeys[ai]); rVals.push_back(aVals[ai]); ++ai; ++bi; } // A wins on tie
    }
    while (ai < aKeys.size()) { rKeys.push_back(aKeys[ai]); rVals.push_back(aVals[ai]); ++ai; }
    while (bi < bKeys.size()) { rKeys.push_back(bKeys[bi]); rVals.push_back(bVals[bi]); ++bi; }
    return {rKeys, rVals};
}

static KVResult cpuSetSymDifferenceByKey(const std::vector<uint32_t>& aKeys,
                                         const std::vector<uint32_t>& aVals,
                                         const std::vector<uint32_t>& bKeys,
                                         const std::vector<uint32_t>& bVals) {
    std::vector<uint32_t> rKeys, rVals;
    size_t ai = 0, bi = 0;
    while (ai < aKeys.size() && bi < bKeys.size()) {
        if (aKeys[ai] < bKeys[bi])      { rKeys.push_back(aKeys[ai]); rVals.push_back(aVals[ai]); ++ai; }
        else if (aKeys[ai] > bKeys[bi]) { rKeys.push_back(bKeys[bi]); rVals.push_back(bVals[bi]); ++bi; }
        else { ++ai; ++bi; }
    }
    while (ai < aKeys.size()) { rKeys.push_back(aKeys[ai]); rVals.push_back(aVals[ai]); ++ai; }
    while (bi < bKeys.size()) { rKeys.push_back(bKeys[bi]); rVals.push_back(bVals[bi]); ++bi; }
    return {rKeys, rVals};
}

static KVResult cpuReferenceByKey(const std::vector<uint32_t>& aKeys,
                                  const std::vector<uint32_t>& aVals,
                                  const std::vector<uint32_t>& bKeys,
                                  const std::vector<uint32_t>& bVals,
                                  uint32_t opMode) {
    switch (opMode) {
        case 0: return cpuSetIntersectionByKey(aKeys, aVals, bKeys, bVals);
        case 1: return cpuSetDifferenceByKey(aKeys, aVals, bKeys, bVals);
        case 2: return cpuSetUnionByKey(aKeys, aVals, bKeys, bVals);
        case 3: return cpuSetSymDifferenceByKey(aKeys, aVals, bKeys, bVals);
        default: return {{}, {}};
    }
}

// ---------------------------------------------------------------------------
// Max output size per operation
// ---------------------------------------------------------------------------
static uint32_t getMaxOutputSize(uint32_t opMode, uint32_t aLen, uint32_t bLen) {
    switch (opMode) {
        case 0: return std::min(aLen, bLen);
        case 1: return aLen;
        case 2: return aLen + bLen;
        case 3: return aLen + bLen;
        default: return aLen + bLen;
    }
}

// ---------------------------------------------------------------------------
// Global state for async callbacks
// ---------------------------------------------------------------------------
static WGPUAdapter g_adapter = nullptr;
static WGPUDevice  g_device  = nullptr;
static bool        g_requestDone = false;
static uint32_t    g_subgroupSize = 32;

static void onAdapterReady(WGPURequestAdapterStatus status,
                           WGPUAdapter adapter,
                           WGPUStringView message,
                           void*, void*) {
    if (status != WGPURequestAdapterStatus_Success) {
        fprintf(stderr, "Adapter request failed: %.*s\n",
                (int)message.length, message.data ? message.data : "");
        exit(1);
    }
    g_adapter = adapter;
    g_requestDone = true;
}

static void onDeviceReady(WGPURequestDeviceStatus status,
                          WGPUDevice device,
                          WGPUStringView message,
                          void*, void*) {
    if (status != WGPURequestDeviceStatus_Success) {
        fprintf(stderr, "Device request failed: %.*s\n",
                (int)message.length, message.data ? message.data : "");
        exit(1);
    }
    g_device = device;
    g_requestDone = true;
}

static void onDeviceError(WGPUDevice const*, WGPUErrorType type,
                          WGPUStringView message,
                          void*, void*) {
    fprintf(stderr, "[WebGPU Error %u] %.*s\n", (unsigned)type,
            (int)message.length, message.data ? message.data : "");
}

static bool g_mapDone = false;
static void onMapDone(WGPUMapAsyncStatus status, WGPUStringView message,
                      void*, void*) {
    if (status != WGPUMapAsyncStatus_Success) {
        fprintf(stderr, "Buffer map failed: %.*s\n",
                (int)message.length, message.data ? message.data : "");
        exit(1);
    }
    g_mapDone = true;
}

static WGPUBufferMapCallbackInfo mapCallbackInfo() {
    WGPUBufferMapCallbackInfo info{};
    info.mode = WGPUCallbackMode_AllowProcessEvents;
    info.callback = onMapDone;
    return info;
}

// ---------------------------------------------------------------------------
// Bind group entry helper
// ---------------------------------------------------------------------------
static WGPUBindGroupEntry bufEntry(uint32_t binding, WGPUBuffer buf) {
    WGPUBindGroupEntry e{};
    e.binding = binding;
    e.buffer = buf;
    e.offset = 0;
    e.size = WGPU_WHOLE_SIZE;
    return e;
}

// ---------------------------------------------------------------------------
// GPU Validation: run pipeline once, read back and compare with CPU
// ---------------------------------------------------------------------------
static bool runValidation(
    WGPUDevice device, WGPUQueue queue,
    WGPUComputePipeline diagPipeline, WGPUBindGroupLayout diagBGL,
    WGPUComputePipeline lookbackPipeline, WGPUBindGroupLayout lookbackBGL,
    const std::vector<uint32_t>& aKeys,
    const std::vector<uint32_t>& aVals,
    const std::vector<uint32_t>& bKeys,
    const std::vector<uint32_t>& bVals,
    uint32_t opMode)
{
    const uint32_t a_len = (uint32_t)aKeys.size();
    const uint32_t b_len = (uint32_t)bKeys.size();
    const uint32_t total = a_len + b_len;
    const uint32_t numWg = (total + NV - 1) / NV;
    const uint32_t maxOutput = getMaxOutputSize(opMode, a_len, b_len);

    const uint32_t subgroupsPerWg = DPI_WORKGROUP_SIZE / g_subgroupSize;
    const uint32_t dpiBlocks = (numWg + subgroupsPerWg - 1) / subgroupsPerWg;
    const uint32_t dpiDispatchX = std::min(dpiBlocks, MAX_DISPATCH_X);
    const uint32_t dpiDispatchY = (dpiBlocks + MAX_DISPATCH_X - 1) / MAX_DISPATCH_X;
    const uint32_t lbDispatchX = std::min(numWg, MAX_DISPATCH_X);
    const uint32_t lbDispatchY = (numWg + MAX_DISPATCH_X - 1) / MAX_DISPATCH_X;

    auto makeBuf = [&](const char* label, uint64_t size, WGPUBufferUsage usage) {
        WGPUBufferDescriptor desc{};
        desc.label = wgpuStr(label);
        desc.size = std::max<uint64_t>(size, 4);
        desc.usage = usage;
        return wgpuDeviceCreateBuffer(device, &desc);
    };

    // 4 input buffers (keys + values for A and B)
    WGPUBuffer bufAKeys    = makeBuf("AKeys",     (uint64_t)a_len * 4, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufAVals    = makeBuf("AVals",     (uint64_t)a_len * 4, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufBKeys    = makeBuf("BKeys",     (uint64_t)b_len * 4, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufBVals    = makeBuf("BVals",     (uint64_t)b_len * 4, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);

    WGPUBuffer bufALen     = makeBuf("aLen",      4,                                    WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufBLen     = makeBuf("bLen",      4,                                    WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufNumWg    = makeBuf("numWg",     4,                                    WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufDPI      = makeBuf("DPI",       (uint64_t)(2*(numWg+1))*4,            WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    WGPUBuffer bufState    = makeBuf("State",     (uint64_t)numWg * 4,                  WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);

    // 2 output buffers (keys + values)
    uint64_t outputBytes   = (uint64_t)std::max(maxOutput, 1u) * 4;
    WGPUBuffer bufOutputKeys = makeBuf("OutputKeys", outputBytes,                       WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    WGPUBuffer bufOutputVals = makeBuf("OutputVals", outputBytes,                       WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    WGPUBuffer bufTotalCnt   = makeBuf("TotalCount", 4,                                 WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst);

    // Staging buffers
    WGPUBuffer stagingCnt     = makeBuf("stagingCnt",     4,            WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
    WGPUBuffer stagingOutKeys = makeBuf("stagingOutKeys", outputBytes,  WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
    WGPUBuffer stagingOutVals = makeBuf("stagingOutVals", outputBytes,  WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);

    // Upload
    wgpuQueueWriteBuffer(queue, bufAKeys, 0, aKeys.data(), a_len * 4);
    wgpuQueueWriteBuffer(queue, bufAVals, 0, aVals.data(), a_len * 4);
    wgpuQueueWriteBuffer(queue, bufBKeys, 0, bKeys.data(), b_len * 4);
    wgpuQueueWriteBuffer(queue, bufBVals, 0, bVals.data(), b_len * 4);
    wgpuQueueWriteBuffer(queue, bufALen,  0, &a_len, 4);
    wgpuQueueWriteBuffer(queue, bufBLen,  0, &b_len, 4);
    wgpuQueueWriteBuffer(queue, bufNumWg, 0, &numWg, 4);

    std::vector<uint32_t> zeros(numWg, 0);
    uint32_t zero = 0;
    wgpuQueueWriteBuffer(queue, bufState,    0, zeros.data(), numWg * 4);
    wgpuQueueWriteBuffer(queue, bufTotalCnt, 0, &zero, 4);

    // DPI bind group: keys only (bindings 0,1 = A keys, B keys)
    WGPUBindGroupEntry diagEntries[] = {
        bufEntry(0, bufAKeys), bufEntry(1, bufBKeys), bufEntry(2, bufDPI),
        bufEntry(3, bufALen), bufEntry(4, bufBLen), bufEntry(5, bufNumWg),
    };
    WGPUBindGroupDescriptor diagBGD{};
    diagBGD.layout = diagBGL;
    diagBGD.entryCount = 6;
    diagBGD.entries = diagEntries;
    WGPUBindGroup diagBG = wgpuDeviceCreateBindGroup(device, &diagBGD);

    // Lookback bind group: 12 entries for by-key variant
    WGPUBindGroupEntry lbEntries[] = {
        bufEntry(0,  bufAKeys),
        bufEntry(1,  bufAVals),
        bufEntry(2,  bufBKeys),
        bufEntry(3,  bufBVals),
        bufEntry(4,  bufDPI),
        bufEntry(5,  bufState),
        bufEntry(6,  bufOutputKeys),
        bufEntry(7,  bufOutputVals),
        bufEntry(8,  bufTotalCnt),
        bufEntry(9,  bufALen),
        bufEntry(10, bufBLen),
        bufEntry(11, bufNumWg),
    };
    WGPUBindGroupDescriptor lbBGD{};
    lbBGD.layout = lookbackBGL;
    lbBGD.entryCount = 12;
    lbBGD.entries = lbEntries;
    WGPUBindGroup lbBG = wgpuDeviceCreateBindGroup(device, &lbBGD);

    // Encode
    WGPUCommandEncoderDescriptor encDesc{};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &encDesc);

    {
        WGPUComputePassDescriptor cpd{};
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &cpd);
        wgpuComputePassEncoderSetPipeline(pass, diagPipeline);
        wgpuComputePassEncoderSetBindGroup(pass, 0, diagBG, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, dpiDispatchX, dpiDispatchY, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    }
    {
        WGPUComputePassDescriptor cpd{};
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &cpd);
        wgpuComputePassEncoderSetPipeline(pass, lookbackPipeline);
        wgpuComputePassEncoderSetBindGroup(pass, 0, lbBG, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, lbDispatchX, lbDispatchY, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    }

    wgpuCommandEncoderCopyBufferToBuffer(encoder, bufTotalCnt,   0, stagingCnt,     0, 4);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, bufOutputKeys, 0, stagingOutKeys, 0, outputBytes);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, bufOutputVals, 0, stagingOutVals, 0, outputBytes);

    WGPUCommandBufferDescriptor cbDesc{};
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(encoder, &cbDesc);
    wgpuQueueSubmit(queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(encoder);

    // Read back count
    g_mapDone = false;
    wgpuBufferMapAsync(stagingCnt, WGPUMapMode_Read, 0, 4, mapCallbackInfo());
    while (!g_mapDone) { wgpuDevicePoll(device, true, nullptr); }
    uint32_t gpuCount = *(const uint32_t*)wgpuBufferGetConstMappedRange(stagingCnt, 0, 4);
    wgpuBufferUnmap(stagingCnt);

    // Read back output keys
    g_mapDone = false;
    wgpuBufferMapAsync(stagingOutKeys, WGPUMapMode_Read, 0, outputBytes, mapCallbackInfo());
    while (!g_mapDone) { wgpuDevicePoll(device, true, nullptr); }
    const uint32_t* gpuKeysData = (const uint32_t*)wgpuBufferGetConstMappedRange(stagingOutKeys, 0, outputBytes);
    std::vector<uint32_t> gpuKeys(gpuKeysData, gpuKeysData + gpuCount);
    wgpuBufferUnmap(stagingOutKeys);

    // Read back output values
    g_mapDone = false;
    wgpuBufferMapAsync(stagingOutVals, WGPUMapMode_Read, 0, outputBytes, mapCallbackInfo());
    while (!g_mapDone) { wgpuDevicePoll(device, true, nullptr); }
    const uint32_t* gpuValsData = (const uint32_t*)wgpuBufferGetConstMappedRange(stagingOutVals, 0, outputBytes);
    std::vector<uint32_t> gpuVals(gpuValsData, gpuValsData + gpuCount);
    wgpuBufferUnmap(stagingOutVals);

    auto cpuResult = cpuReferenceByKey(aKeys, aVals, bKeys, bVals, opMode);
    bool keysPass = (gpuKeys == cpuResult.first);
    bool valsPass = (gpuVals == cpuResult.second);
    bool pass = keysPass && valsPass;

    printf("  %-16s: GPU=%u  CPU=%zu  keys=%s  vals=%s  %s\n",
           OP_NAMES[opMode], gpuCount, cpuResult.first.size(),
           keysPass ? "OK" : "FAIL",
           valsPass ? "OK" : "FAIL",
           pass ? "PASS" : "FAIL");

    // Cleanup
    wgpuBindGroupRelease(diagBG);
    wgpuBindGroupRelease(lbBG);
    wgpuBufferRelease(bufAKeys);
    wgpuBufferRelease(bufAVals);
    wgpuBufferRelease(bufBKeys);
    wgpuBufferRelease(bufBVals);
    wgpuBufferRelease(bufALen);
    wgpuBufferRelease(bufBLen);
    wgpuBufferRelease(bufNumWg);
    wgpuBufferRelease(bufDPI);
    wgpuBufferRelease(bufState);
    wgpuBufferRelease(bufOutputKeys);
    wgpuBufferRelease(bufOutputVals);
    wgpuBufferRelease(bufTotalCnt);
    wgpuBufferRelease(stagingCnt);
    wgpuBufferRelease(stagingOutKeys);
    wgpuBufferRelease(stagingOutVals);

    return pass;
}

// ---------------------------------------------------------------------------
// GPU Timestamp Benchmark
// ---------------------------------------------------------------------------
struct BenchmarkTiming {
    double dpiMs;
    double lookbackMs;
    double totalMs;
};

static BenchmarkTiming runBenchmark(
    WGPUDevice device, WGPUQueue queue,
    WGPUComputePipeline diagPipeline, WGPUBindGroupLayout diagBGL,
    WGPUComputePipeline lookbackPipeline, WGPUBindGroupLayout lookbackBGL,
    const std::vector<uint32_t>& aKeys,
    const std::vector<uint32_t>& aVals,
    const std::vector<uint32_t>& bKeys,
    const std::vector<uint32_t>& bVals,
    uint32_t opMode,
    int warmupIters, int measuredIters)
{
    const uint32_t a_len = (uint32_t)aKeys.size();
    const uint32_t b_len = (uint32_t)bKeys.size();
    const uint32_t total = a_len + b_len;
    const uint32_t numWg = (total + NV - 1) / NV;
    const uint32_t maxOutput = getMaxOutputSize(opMode, a_len, b_len);

    const uint32_t subgroupsPerWg = DPI_WORKGROUP_SIZE / g_subgroupSize;
    const uint32_t dpiBlocks = (numWg + subgroupsPerWg - 1) / subgroupsPerWg;
    const uint32_t dpiDispatchX = std::min(dpiBlocks, MAX_DISPATCH_X);
    const uint32_t dpiDispatchY = (dpiBlocks + MAX_DISPATCH_X - 1) / MAX_DISPATCH_X;
    const uint32_t lbDispatchX = std::min(numWg, MAX_DISPATCH_X);
    const uint32_t lbDispatchY = (numWg + MAX_DISPATCH_X - 1) / MAX_DISPATCH_X;

    auto makeBuf = [&](const char* label, uint64_t size, WGPUBufferUsage usage) {
        WGPUBufferDescriptor desc{};
        desc.label = wgpuStr(label);
        desc.size = std::max<uint64_t>(size, 4);
        desc.usage = usage;
        return wgpuDeviceCreateBuffer(device, &desc);
    };

    // 4 input buffers
    WGPUBuffer bufAKeys    = makeBuf("AKeys",     (uint64_t)a_len * 4, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufAVals    = makeBuf("AVals",     (uint64_t)a_len * 4, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufBKeys    = makeBuf("BKeys",     (uint64_t)b_len * 4, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufBVals    = makeBuf("BVals",     (uint64_t)b_len * 4, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);

    WGPUBuffer bufALen     = makeBuf("aLen",      4,                                     WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufBLen     = makeBuf("bLen",      4,                                     WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufNumWg    = makeBuf("numWg",     4,                                     WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufDPI      = makeBuf("DPI",       (uint64_t)(2*(numWg+1))*4,             WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    WGPUBuffer bufState    = makeBuf("State",     (uint64_t)numWg * 4,                   WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);

    // 2 output buffers
    uint64_t outputBytes = (uint64_t)std::max(maxOutput, 1u) * 4;
    WGPUBuffer bufOutputKeys = makeBuf("OutputKeys", outputBytes,                        WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    WGPUBuffer bufOutputVals = makeBuf("OutputVals", outputBytes,                        WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    WGPUBuffer bufTotalCnt   = makeBuf("TotalCount", 4,                                  WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst);

    // Upload data
    wgpuQueueWriteBuffer(queue, bufAKeys, 0, aKeys.data(), a_len * 4);
    wgpuQueueWriteBuffer(queue, bufAVals, 0, aVals.data(), a_len * 4);
    wgpuQueueWriteBuffer(queue, bufBKeys, 0, bKeys.data(), b_len * 4);
    wgpuQueueWriteBuffer(queue, bufBVals, 0, bVals.data(), b_len * 4);
    wgpuQueueWriteBuffer(queue, bufALen,  0, &a_len, 4);
    wgpuQueueWriteBuffer(queue, bufBLen,  0, &b_len, 4);
    wgpuQueueWriteBuffer(queue, bufNumWg, 0, &numWg, 4);

    // DPI bind group: keys only
    WGPUBindGroupEntry diagEntries[] = {
        bufEntry(0, bufAKeys), bufEntry(1, bufBKeys), bufEntry(2, bufDPI),
        bufEntry(3, bufALen), bufEntry(4, bufBLen), bufEntry(5, bufNumWg),
    };
    WGPUBindGroupDescriptor diagBGD{};
    diagBGD.label = wgpuStr("DPI BG");
    diagBGD.layout = diagBGL;
    diagBGD.entryCount = 6;
    diagBGD.entries = diagEntries;
    WGPUBindGroup diagBG = wgpuDeviceCreateBindGroup(device, &diagBGD);

    // Lookback bind group: 12 entries
    WGPUBindGroupEntry lbEntries[] = {
        bufEntry(0,  bufAKeys),
        bufEntry(1,  bufAVals),
        bufEntry(2,  bufBKeys),
        bufEntry(3,  bufBVals),
        bufEntry(4,  bufDPI),
        bufEntry(5,  bufState),
        bufEntry(6,  bufOutputKeys),
        bufEntry(7,  bufOutputVals),
        bufEntry(8,  bufTotalCnt),
        bufEntry(9,  bufALen),
        bufEntry(10, bufBLen),
        bufEntry(11, bufNumWg),
    };
    WGPUBindGroupDescriptor lbBGD{};
    lbBGD.label = wgpuStr("LB BG");
    lbBGD.layout = lookbackBGL;
    lbBGD.entryCount = 12;
    lbBGD.entries = lbEntries;
    WGPUBindGroup lbBG = wgpuDeviceCreateBindGroup(device, &lbBGD);

    // Timestamp queries
    WGPUQuerySetDescriptor qsDesc{};
    qsDesc.type = WGPUQueryType_Timestamp;
    qsDesc.count = 4;
    WGPUQuerySet querySet = wgpuDeviceCreateQuerySet(device, &qsDesc);

    const uint64_t tsBufferSize = 4 * sizeof(uint64_t);
    WGPUBuffer tsResolveBuf = makeBuf("tsResolve", tsBufferSize,
                                     WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc);
    WGPUBuffer tsMapBuf     = makeBuf("tsMap",     tsBufferSize,
                                     WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);

    std::vector<uint32_t> zeros(numWg, 0);
    uint32_t zero = 0;

    double dpiSum = 0, lookbackSum = 0, totalSum = 0;
    const int totalIters = warmupIters + measuredIters;

    for (int iter = 0; iter < totalIters; iter++) {
        wgpuQueueWriteBuffer(queue, bufState,    0, zeros.data(), numWg * 4);
        wgpuQueueWriteBuffer(queue, bufTotalCnt, 0, &zero, 4);

        WGPUCommandEncoderDescriptor encDesc{};
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &encDesc);

        {
            WGPUComputePassTimestampWrites tsWrites{};
            tsWrites.querySet = querySet;
            tsWrites.beginningOfPassWriteIndex = 0;
            tsWrites.endOfPassWriteIndex = 1;

            WGPUComputePassDescriptor cpd{};
            cpd.timestampWrites = &tsWrites;
            WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &cpd);
            wgpuComputePassEncoderSetPipeline(pass, diagPipeline);
            wgpuComputePassEncoderSetBindGroup(pass, 0, diagBG, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, dpiDispatchX, dpiDispatchY, 1);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }

        {
            WGPUComputePassTimestampWrites tsWrites{};
            tsWrites.querySet = querySet;
            tsWrites.beginningOfPassWriteIndex = 2;
            tsWrites.endOfPassWriteIndex = 3;

            WGPUComputePassDescriptor cpd{};
            cpd.timestampWrites = &tsWrites;
            WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &cpd);
            wgpuComputePassEncoderSetPipeline(pass, lookbackPipeline);
            wgpuComputePassEncoderSetBindGroup(pass, 0, lbBG, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, lbDispatchX, lbDispatchY, 1);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }

        wgpuCommandEncoderResolveQuerySet(encoder, querySet, 0, 4, tsResolveBuf, 0);
        wgpuCommandEncoderCopyBufferToBuffer(encoder, tsResolveBuf, 0, tsMapBuf, 0, tsBufferSize);

        WGPUCommandBufferDescriptor cbDesc{};
        WGPUCommandBuffer cb = wgpuCommandEncoderFinish(encoder, &cbDesc);
        wgpuQueueSubmit(queue, 1, &cb);
        wgpuCommandBufferRelease(cb);
        wgpuCommandEncoderRelease(encoder);

        g_mapDone = false;
        wgpuBufferMapAsync(tsMapBuf, WGPUMapMode_Read, 0, tsBufferSize, mapCallbackInfo());
        while (!g_mapDone) { wgpuDevicePoll(device, true, nullptr); }

        const uint64_t* ts = (const uint64_t*)wgpuBufferGetConstMappedRange(tsMapBuf, 0, tsBufferSize);
        if (ts && iter >= warmupIters) {
            double dpiNs      = (double)(ts[1] - ts[0]);
            double lookbackNs = (double)(ts[3] - ts[2]);
            double totalNs    = (double)(ts[3] - ts[0]);

            dpiSum      += dpiNs      / 1e6;
            lookbackSum += lookbackNs / 1e6;
            totalSum    += totalNs    / 1e6;
        }
        wgpuBufferUnmap(tsMapBuf);
    }

    wgpuQuerySetDestroy(querySet);
    wgpuQuerySetRelease(querySet);
    wgpuBindGroupRelease(diagBG);
    wgpuBindGroupRelease(lbBG);
    wgpuBufferRelease(bufAKeys);
    wgpuBufferRelease(bufAVals);
    wgpuBufferRelease(bufBKeys);
    wgpuBufferRelease(bufBVals);
    wgpuBufferRelease(bufALen);
    wgpuBufferRelease(bufBLen);
    wgpuBufferRelease(bufNumWg);
    wgpuBufferRelease(bufDPI);
    wgpuBufferRelease(bufState);
    wgpuBufferRelease(bufOutputKeys);
    wgpuBufferRelease(bufOutputVals);
    wgpuBufferRelease(bufTotalCnt);
    wgpuBufferRelease(tsResolveBuf);
    wgpuBufferRelease(tsMapBuf);

    return { dpiSum / measuredIters, lookbackSum / measuredIters, totalSum / measuredIters };
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    printf("=== Unified Set Operations By Key (Balanced Path + Decoupled Lookback) ===\n");
    printf("=== C++ / wgpu-native v24 port (subgroup-optimized DPI) ===\n\n");

    // ---- Instance ----
    WGPUInstanceDescriptor instDesc{};
    WGPUInstance instance = wgpuCreateInstance(&instDesc);
    if (!instance) {
        fprintf(stderr, "Failed to create WebGPU instance\n");
        return 1;
    }

    // ---- Adapter ----
    WGPURequestAdapterOptions adapterOpts{};
    adapterOpts.powerPreference = WGPUPowerPreference_HighPerformance;

    g_requestDone = false;
    WGPURequestAdapterCallbackInfo adapterCbInfo{};
    adapterCbInfo.mode = WGPUCallbackMode_AllowProcessEvents;
    adapterCbInfo.callback = onAdapterReady;
    wgpuInstanceRequestAdapter(instance, &adapterOpts, adapterCbInfo);
    while (!g_requestDone) {
        wgpuInstanceProcessEvents(instance);
    }
    assert(g_adapter);

    WGPUAdapterInfo adapterInfo{};
    wgpuAdapterGetInfo(g_adapter, &adapterInfo);
    printf("Adapter: %.*s\n", (int)adapterInfo.device.length,
           adapterInfo.device.data ? adapterInfo.device.data : "unknown");
    printf("Driver:  %.*s\n", (int)adapterInfo.description.length,
           adapterInfo.description.data ? adapterInfo.description.data : "unknown");
    printf("Backend: ");
    switch (adapterInfo.backendType) {
        case WGPUBackendType_Vulkan: printf("Vulkan\n"); break;
        case WGPUBackendType_D3D12:  printf("D3D12\n");  break;
        case WGPUBackendType_D3D11:  printf("D3D11\n");  break;
        case WGPUBackendType_Metal:  printf("Metal\n");  break;
        default: printf("Other (%d)\n", adapterInfo.backendType);
    }
    printf("\n");

    // ---- Adapter limits ----
    WGPULimits adapterLimits{};
    adapterLimits.nextInChain = nullptr;
    auto retA = wgpuAdapterGetLimits(g_adapter, &adapterLimits);
#if defined(WGPUStatus_Success)
    if (retA != WGPUStatus_Success) {
        fprintf(stderr, "wgpuAdapterGetLimits failed (status=%d)\n", (int)retA);
        return 1;
    }
#else
    if (!retA) {
        fprintf(stderr, "wgpuAdapterGetLimits failed\n");
        return 1;
    }
#endif

    printf("Adapter supported limits:\n");
    printf("  maxBufferSize               = %llu MiB\n",
           (unsigned long long)(adapterLimits.maxBufferSize / (1024ull * 1024ull)));
    printf("  maxStorageBufferBindingSize = %llu MiB\n",
           (unsigned long long)(adapterLimits.maxStorageBufferBindingSize / (1024ull * 1024ull)));
    printf("  maxUniformBufferBindingSize = %llu KiB\n",
           (unsigned long long)(adapterLimits.maxUniformBufferBindingSize / 1024ull));
    printf("\n");

    // ---- Device ----
    WGPUDeviceDescriptor devDesc{};
    devDesc.label = wgpuStr("GPU Device");
    devDesc.requiredLimits = &adapterLimits;
    devDesc.defaultQueue.label = wgpuStr("Default Queue");
    devDesc.uncapturedErrorCallbackInfo.callback = onDeviceError;

    WGPUBool hasTimestampQuery = wgpuAdapterHasFeature(g_adapter, WGPUFeatureName_TimestampQuery);
    WGPUBool hasSubgroups = wgpuAdapterHasFeature(g_adapter,
        (WGPUFeatureName)WGPUNativeFeature_Subgroup);

    std::vector<WGPUFeatureName> requiredFeatures;
    if (hasTimestampQuery) requiredFeatures.push_back(WGPUFeatureName_TimestampQuery);
    if (hasSubgroups)      requiredFeatures.push_back((WGPUFeatureName)WGPUNativeFeature_Subgroup);

    if (!requiredFeatures.empty()) {
        devDesc.requiredFeatureCount = (uint32_t)requiredFeatures.size();
        devDesc.requiredFeatures = requiredFeatures.data();
    }

    g_requestDone = false;
    WGPURequestDeviceCallbackInfo deviceCbInfo{};
    deviceCbInfo.mode = WGPUCallbackMode_AllowProcessEvents;
    deviceCbInfo.callback = onDeviceReady;
    wgpuAdapterRequestDevice(g_adapter, &devDesc, deviceCbInfo);
    while (!g_requestDone) {
        wgpuInstanceProcessEvents(instance);
    }
    assert(g_device);

    WGPUQueue queue = wgpuDeviceGetQueue(g_device);

    // ---- Device limits ----
    WGPULimits deviceLimits{};
    deviceLimits.nextInChain = nullptr;
    auto retD = wgpuDeviceGetLimits(g_device, &deviceLimits);

    #if defined(WGPUStatus_Success)
        if (retD != WGPUStatus_Success) {
            fprintf(stderr, "wgpuDeviceGetLimits failed (status=%d)\n", (int)retD);
            return 1;
        }
    #else
        if (!retD) {
            fprintf(stderr, "wgpuDeviceGetLimits failed\n");
            return 1;
        }
    #endif

    uint64_t maxBufSize = deviceLimits.maxBufferSize;
    uint64_t maxStorageBind = deviceLimits.maxStorageBufferBindingSize;

    g_subgroupSize = 32;

    printf("Device ready.\n");
    printf("  device.maxBufferSize               = %llu MiB\n",
           (unsigned long long)(maxBufSize / (1024ull * 1024ull)));
    printf("  device.maxStorageBufferBindingSize = %llu MiB\n",
           (unsigned long long)(maxStorageBind / (1024ull * 1024ull)));
    printf("Timestamp queries: %s\n", hasTimestampQuery ? "supported" : "NOT supported");
    printf("Subgroups: %s (assumed size=%u, DPI subgroups/wg=%u)\n\n",
           hasSubgroups ? "supported" : "NOT supported",
           g_subgroupSize, DPI_WORKGROUP_SIZE / g_subgroupSize);

    // ---- Load shaders ----
    const char* shaderDirs[] = { "shaders", "../shaders", "../../shaders" };
    const char* shaderDir = nullptr;
    for (auto d : shaderDirs) {
        char probe[256];
        snprintf(probe, sizeof(probe), "%s/balanced_path_biased.wgsl", d);
        std::ifstream test(probe);
        if (test.good()) { shaderDir = d; break; }
    }
    if (!shaderDir) {
        fprintf(stderr, "ERROR: shader directory not found. Run from common_cpp/by_key/ or build/Debug/\n");
        return 1;
    }
    char dpiPath[256], lbPath[256];
    snprintf(dpiPath, sizeof(dpiPath), "%s/balanced_path_biased.wgsl", shaderDir);
    snprintf(lbPath,  sizeof(lbPath),  "%s/set_availability_decoupled_lookback_by_key.wgsl", shaderDir);
    std::string dpiCode        = readFile(dpiPath);
    std::string lbCodeTemplate = readFile(lbPath);

    auto createShaderModule = [&](const char* label, const std::string& code) {
        WGPUShaderSourceWGSL wgslDesc{};
        wgslDesc.chain.sType = WGPUSType_ShaderSourceWGSL;
        wgslDesc.code = {code.c_str(), code.size()};

        WGPUShaderModuleDescriptor smDesc{};
        smDesc.nextInChain = &wgslDesc.chain;
        smDesc.label = wgpuStr(label);
        return wgpuDeviceCreateShaderModule(g_device, &smDesc);
    };

    WGPUShaderModule dpiShaderModule = createShaderModule("DPI Shader", dpiCode);

    // ---- Create bind group layouts ----
    auto bglEntry = [](uint32_t binding, WGPUBufferBindingType type) {
        WGPUBindGroupLayoutEntry e{};
        e.binding = binding;
        e.visibility = WGPUShaderStage_Compute;
        e.buffer.type = type;
        return e;
    };

    // DPI BGL: 6 bindings (keys only, same as not_by_key)
    WGPUBindGroupLayoutEntry diagLayoutEntries[] = {
        bglEntry(0, WGPUBufferBindingType_ReadOnlyStorage),  // a_keys
        bglEntry(1, WGPUBufferBindingType_ReadOnlyStorage),  // b_keys
        bglEntry(2, WGPUBufferBindingType_Storage),          // dpi
        bglEntry(3, WGPUBufferBindingType_Uniform),          // a_length
        bglEntry(4, WGPUBufferBindingType_Uniform),          // b_length
        bglEntry(5, WGPUBufferBindingType_Uniform),          // num_wg
    };
    WGPUBindGroupLayoutDescriptor diagBGLD{};
    diagBGLD.label = wgpuStr("DPI BGL");
    diagBGLD.entryCount = 6;
    diagBGLD.entries = diagLayoutEntries;
    WGPUBindGroupLayout diagBGL = wgpuDeviceCreateBindGroupLayout(g_device, &diagBGLD);

    // Lookback BGL: 12 bindings (by-key variant)
    WGPUBindGroupLayoutEntry lbLayoutEntries[] = {
        bglEntry(0,  WGPUBufferBindingType_ReadOnlyStorage),  // a_keys
        bglEntry(1,  WGPUBufferBindingType_ReadOnlyStorage),  // a_values
        bglEntry(2,  WGPUBufferBindingType_ReadOnlyStorage),  // b_keys
        bglEntry(3,  WGPUBufferBindingType_ReadOnlyStorage),  // b_values
        bglEntry(4,  WGPUBufferBindingType_ReadOnlyStorage),  // dpi
        bglEntry(5,  WGPUBufferBindingType_Storage),          // state
        bglEntry(6,  WGPUBufferBindingType_Storage),          // output_keys
        bglEntry(7,  WGPUBufferBindingType_Storage),          // output_values
        bglEntry(8,  WGPUBufferBindingType_Storage),          // total_count
        bglEntry(9,  WGPUBufferBindingType_Uniform),          // a_length
        bglEntry(10, WGPUBufferBindingType_Uniform),          // b_length
        bglEntry(11, WGPUBufferBindingType_Uniform),          // num_wg_total
    };
    WGPUBindGroupLayoutDescriptor lbBGLD{};
    lbBGLD.label = wgpuStr("Lookback BGL");
    lbBGLD.entryCount = 12;
    lbBGLD.entries = lbLayoutEntries;
    WGPUBindGroupLayout lookbackBGL = wgpuDeviceCreateBindGroupLayout(g_device, &lbBGLD);

    auto makePipeline = [&](const char* label, WGPUBindGroupLayout bgl,
                            WGPUShaderModule sm, const char* entryPoint) {
        WGPUPipelineLayoutDescriptor plDesc{};
        plDesc.bindGroupLayoutCount = 1;
        plDesc.bindGroupLayouts = &bgl;
        WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(g_device, &plDesc);

        WGPUComputePipelineDescriptor cpDesc{};
        cpDesc.label = wgpuStr(label);
        cpDesc.layout = pl;
        cpDesc.compute.module = sm;
        cpDesc.compute.entryPoint = wgpuStr(entryPoint);

        WGPUComputePipeline pipe = wgpuDeviceCreateComputePipeline(g_device, &cpDesc);
        wgpuPipelineLayoutRelease(pl);
        return pipe;
    };

    WGPUComputePipeline diagPipeline = makePipeline("DPI Pipeline", diagBGL,
                                                    dpiShaderModule, "compute_diagonals");

    printf("DPI pipeline created.\n");

    // ======================================================================
    // Locate data directory
    // ======================================================================
    const char* dataDirs[] = {
        "../../../../public/data",
        "../../../public/data",
        "../../public/data",
        "../public/data",
        "public/data",
    };
    const char* dataDir = nullptr;
    for (auto d : dataDirs) {
        char probe[256];
        snprintf(probe, sizeof(probe), "%s/A_1e2.bin", d);
        std::ifstream test(probe);
        if (test.good()) { dataDir = d; break; }
    }

    // ======================================================================
    // Loop over all 4 operations
    // ======================================================================
    for (uint32_t opMode = 0; opMode < 4; opMode++) {
        printf("\n========================================\n");
        printf("  Set %s By Key (OP_MODE=%u)\n", OP_NAMES[opMode], opMode);
        printf("========================================\n");

        std::string lbCode = setOpMode(lbCodeTemplate, opMode);
        WGPUShaderModule lbShaderModule = createShaderModule("Lookback By Key Shader", lbCode);
        WGPUComputePipeline lbPipeline = makePipeline("Lookback By Key Pipeline", lookbackBGL,
                                                      lbShaderModule, "decoupled_lookback_by_key_kernel");

        printf("Lookback by-key pipeline created (OP_MODE=%u).\n", opMode);

        if (dataDir) {
            char pathA[256], pathB[256];
            snprintf(pathA, sizeof(pathA), "%s/A_1e2.bin", dataDir);
            snprintf(pathB, sizeof(pathB), "%s/B_1e2.bin", dataDir);

            std::ifstream fA(pathA), fB(pathB);
            if (fA.good() && fB.good()) {
                fA.close(); fB.close();
                auto A = loadBinaryData(pathA);
                auto B = loadBinaryData(pathB);

                // Generate synthetic values
                auto aVals = generateValues(A.size(), 0);
                auto bVals = generateValues(B.size(), 1000000000u);

                printf("\nValidation (%zuM + %zuM):\n", A.size()/1000000, B.size()/1000000);
                bool ok = runValidation(g_device, queue,
                                        diagPipeline, diagBGL,
                                        lbPipeline, lookbackBGL,
                                        A, aVals, B, bVals, opMode);
                if (!ok) {
                    printf("  ** VALIDATION FAILED — skipping benchmarks for this op **\n");
                    wgpuComputePipelineRelease(lbPipeline);
                    wgpuShaderModuleRelease(lbShaderModule);
                    continue;
                }
            } else {
                printf("\nValidation skipped: 1e2 data files not found.\n");
            }
        }

        if (hasTimestampQuery && dataDir) {
            const int WARMUP = 10;
            const int MEASURED = 100;

            printf("\nBenchmark (%d warmup, %d measured):\n", WARMUP, MEASURED);
            printf("%-10s %12s  %10s  %12s  %10s\n",
                   "Dataset", "Input Size", "DPI(ms)", "Lookback(ms)", "Total(ms)");
            printf("%-10s %12s  %10s  %12s  %10s\n",
                   "-------", "----------", "-------", "------------", "--------");

            const char* bmSizes[]  = {"1", "2", "4", "8", "16", "32", "64", "128"};
            const char* bmRanges[] = {"e2", "e6"};

            for (auto sz : bmSizes) {
                for (auto rng : bmRanges) {
                    char pathA[256], pathB[256], dsName[64];
                    snprintf(pathA, sizeof(pathA), "%s/A_%s%s.bin", dataDir, sz, rng);
                    snprintf(pathB, sizeof(pathB), "%s/B_%s%s.bin", dataDir, sz, rng);
                    snprintf(dsName, sizeof(dsName), "%s%s", sz, rng);

                    std::ifstream fA(pathA), fB(pathB);
                    if (!fA.good() || !fB.good()) continue;
                    fA.close(); fB.close();

                    auto A = loadBinaryData(pathA);
                    auto B = loadBinaryData(pathB);

                    // Generate synthetic values
                    auto aVals = generateValues(A.size(), 0);
                    auto bVals = generateValues(B.size(), 1000000000u);

                    uint32_t maxOut = getMaxOutputSize(opMode, (uint32_t)A.size(), (uint32_t)B.size());
                    // Account for both keys and values buffers (2x per array)
                    uint64_t largestBuf = std::max({
                        (uint64_t)A.size() * 4,
                        (uint64_t)B.size() * 4,
                        (uint64_t)maxOut * 4
                    });

                    // Whole-buffer binding => must respect BOTH limits
                    if (largestBuf > maxBufSize || largestBuf > maxStorageBind) {
                        printf("%-10s %10zuM  (skipped: exceeds limit)\n",
                               dsName, A.size()/1000000);
                        continue;
                    }

                    char inputStr[32];
                    snprintf(inputStr, sizeof(inputStr), "%zuM+%zuM",
                             A.size()/1000000, B.size()/1000000);

                    auto timing = runBenchmark(g_device, queue,
                                               diagPipeline, diagBGL,
                                               lbPipeline, lookbackBGL,
                                               A, aVals, B, bVals,
                                               opMode, WARMUP, MEASURED);

                    printf("%-10s %12s  %10.3f  %12.3f  %10.3f\n",
                           dsName, inputStr,
                           timing.dpiMs, timing.lookbackMs, timing.totalMs);
                }
            }
        } else if (!hasTimestampQuery) {
            printf("\nBenchmark skipped: timestamp queries not supported.\n");
        }

        wgpuComputePipelineRelease(lbPipeline);
        wgpuShaderModuleRelease(lbShaderModule);
    }

    printf("\n=== Done ===\n");

    // Cleanup shared resources
    wgpuComputePipelineRelease(diagPipeline);
    wgpuBindGroupLayoutRelease(diagBGL);
    wgpuBindGroupLayoutRelease(lookbackBGL);
    wgpuShaderModuleRelease(dpiShaderModule);
    wgpuQueueRelease(queue);
    wgpuDeviceRelease(g_device);
    wgpuAdapterRelease(g_adapter);
    wgpuInstanceRelease(instance);

    return 0;
}
