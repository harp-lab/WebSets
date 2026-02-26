/**
 * Unified Set Operations using Balanced Path + Decoupled Lookback (C++ / wgpu-native)
 *
 * Two-phase GPU pipeline:
 *   1. DPI (Diagonal Path Intersection) - compute merge path partition boundaries
 *   2. Decoupled Lookback - single-pass count + scan + write results
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
// CPU reference implementations (sorted multisets)
// ---------------------------------------------------------------------------
static std::vector<uint32_t> cpuSetIntersection(const std::vector<uint32_t>& a,
                                                const std::vector<uint32_t>& b) {
    std::vector<uint32_t> result;
    size_t ai = 0, bi = 0;
    while (ai < a.size() && bi < b.size()) {
        if (a[ai] < b[bi])      { ++ai; }
        else if (a[ai] > b[bi]) { ++bi; }
        else { result.push_back(a[ai]); ++ai; ++bi; }
    }
    return result;
}

static std::vector<uint32_t> cpuSetDifference(const std::vector<uint32_t>& a,
                                              const std::vector<uint32_t>& b) {
    std::vector<uint32_t> result;
    size_t ai = 0, bi = 0;
    while (ai < a.size() && bi < b.size()) {
        if (a[ai] < b[bi])      { result.push_back(a[ai]); ++ai; }
        else if (a[ai] > b[bi]) { ++bi; }
        else { ++ai; ++bi; }
    }
    while (ai < a.size()) { result.push_back(a[ai]); ++ai; }
    return result;
}

static std::vector<uint32_t> cpuSetUnion(const std::vector<uint32_t>& a,
                                         const std::vector<uint32_t>& b) {
    std::vector<uint32_t> result;
    size_t ai = 0, bi = 0;
    while (ai < a.size() && bi < b.size()) {
        if (a[ai] < b[bi])      { result.push_back(a[ai]); ++ai; }
        else if (a[ai] > b[bi]) { result.push_back(b[bi]); ++bi; }
        else { result.push_back(a[ai]); ++ai; ++bi; }
    }
    while (ai < a.size()) { result.push_back(a[ai]); ++ai; }
    while (bi < b.size()) { result.push_back(b[bi]); ++bi; }
    return result;
}

static std::vector<uint32_t> cpuSetSymDifference(const std::vector<uint32_t>& a,
                                                 const std::vector<uint32_t>& b) {
    std::vector<uint32_t> result;
    size_t ai = 0, bi = 0;
    while (ai < a.size() && bi < b.size()) {
        if (a[ai] < b[bi])      { result.push_back(a[ai]); ++ai; }
        else if (a[ai] > b[bi]) { result.push_back(b[bi]); ++bi; }
        else { ++ai; ++bi; }
    }
    while (ai < a.size()) { result.push_back(a[ai]); ++ai; }
    while (bi < b.size()) { result.push_back(b[bi]); ++bi; }
    return result;
}

static std::vector<uint32_t> cpuReference(const std::vector<uint32_t>& a,
                                          const std::vector<uint32_t>& b,
                                          uint32_t opMode) {
    switch (opMode) {
        case 0: return cpuSetIntersection(a, b);
        case 1: return cpuSetDifference(a, b);
        case 2: return cpuSetUnion(a, b);
        case 3: return cpuSetSymDifference(a, b);
        default: return {};
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
//   - Use WGPU_WHOLE_SIZE by default (more robust with different headers)
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
    const std::vector<uint32_t>& setA,
    const std::vector<uint32_t>& setB,
    uint32_t opMode)
{
    const uint32_t a_len = (uint32_t)setA.size();
    const uint32_t b_len = (uint32_t)setB.size();
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

    WGPUBuffer bufA        = makeBuf("A",          (uint64_t)a_len * 4,                  WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufB        = makeBuf("B",          (uint64_t)b_len * 4,                  WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufALen     = makeBuf("aLen",       4,                                    WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufBLen     = makeBuf("bLen",       4,                                    WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufNumWg    = makeBuf("numWg",      4,                                    WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufDPI      = makeBuf("DPI",        (uint64_t)(2*(numWg+1))*4,            WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    WGPUBuffer bufState    = makeBuf("State",      (uint64_t)numWg * 4,                  WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    uint64_t outputBytes   = (uint64_t)std::max(maxOutput, 1u) * 4;
    WGPUBuffer bufOutput   = makeBuf("Output",     outputBytes,                          WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    WGPUBuffer bufTotalCnt = makeBuf("TotalCount", 4,                                    WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst);

    WGPUBuffer stagingCnt  = makeBuf("stagingCnt", 4,            WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
    WGPUBuffer stagingOut  = makeBuf("stagingOut", outputBytes,  WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);

    // Upload
    wgpuQueueWriteBuffer(queue, bufA,     0, setA.data(), a_len * 4);
    wgpuQueueWriteBuffer(queue, bufB,     0, setB.data(), b_len * 4);
    wgpuQueueWriteBuffer(queue, bufALen,  0, &a_len, 4);
    wgpuQueueWriteBuffer(queue, bufBLen,  0, &b_len, 4);
    wgpuQueueWriteBuffer(queue, bufNumWg, 0, &numWg, 4);

    std::vector<uint32_t> zeros(numWg, 0);
    uint32_t zero = 0;
    wgpuQueueWriteBuffer(queue, bufState,    0, zeros.data(), numWg * 4);
    wgpuQueueWriteBuffer(queue, bufTotalCnt, 0, &zero, 4);

    // Bind groups
    WGPUBindGroupEntry diagEntries[] = {
        bufEntry(0, bufA), bufEntry(1, bufB), bufEntry(2, bufDPI),
        bufEntry(3, bufALen), bufEntry(4, bufBLen), bufEntry(5, bufNumWg),
    };
    WGPUBindGroupDescriptor diagBGD{};
    diagBGD.layout = diagBGL;
    diagBGD.entryCount = 6;
    diagBGD.entries = diagEntries;
    WGPUBindGroup diagBG = wgpuDeviceCreateBindGroup(device, &diagBGD);

    WGPUBindGroupEntry lbEntries[] = {
        bufEntry(0, bufA), bufEntry(1, bufB), bufEntry(2, bufDPI),
        bufEntry(3, bufState), bufEntry(4, bufOutput), bufEntry(5, bufTotalCnt),
        bufEntry(6, bufALen), bufEntry(7, bufBLen), bufEntry(8, bufNumWg),
    };
    WGPUBindGroupDescriptor lbBGD{};
    lbBGD.layout = lookbackBGL;
    lbBGD.entryCount = 9;
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

    wgpuCommandEncoderCopyBufferToBuffer(encoder, bufTotalCnt, 0, stagingCnt, 0, 4);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, bufOutput,   0, stagingOut, 0, outputBytes);

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

    // Read back output
    g_mapDone = false;
    wgpuBufferMapAsync(stagingOut, WGPUMapMode_Read, 0, outputBytes, mapCallbackInfo());
    while (!g_mapDone) { wgpuDevicePoll(device, true, nullptr); }
    const uint32_t* gpuData = (const uint32_t*)wgpuBufferGetConstMappedRange(stagingOut, 0, outputBytes);
    std::vector<uint32_t> gpuResult(gpuData, gpuData + gpuCount);
    wgpuBufferUnmap(stagingOut);

    auto cpuResult = cpuReference(setA, setB, opMode);
    bool pass = (gpuResult == cpuResult);

    printf("  %-16s: GPU=%u  CPU=%zu  %s\n",
           OP_NAMES[opMode], gpuCount, cpuResult.size(),
           pass ? "PASS" : "FAIL");

    // Cleanup
    wgpuBindGroupRelease(diagBG);
    wgpuBindGroupRelease(lbBG);
    wgpuBufferRelease(bufA);
    wgpuBufferRelease(bufB);
    wgpuBufferRelease(bufALen);
    wgpuBufferRelease(bufBLen);
    wgpuBufferRelease(bufNumWg);
    wgpuBufferRelease(bufDPI);
    wgpuBufferRelease(bufState);
    wgpuBufferRelease(bufOutput);
    wgpuBufferRelease(bufTotalCnt);
    wgpuBufferRelease(stagingCnt);
    wgpuBufferRelease(stagingOut);

    return pass;
}

// ---------------------------------------------------------------------------
// GPU Timestamp Benchmark (same logic as your original)
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
    const std::vector<uint32_t>& setA,
    const std::vector<uint32_t>& setB,
    uint32_t opMode,
    int warmupIters, int measuredIters)
{
    const uint32_t a_len = (uint32_t)setA.size();
    const uint32_t b_len = (uint32_t)setB.size();
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

    WGPUBuffer bufA        = makeBuf("A",          (uint64_t)a_len * 4,                   WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufB        = makeBuf("B",          (uint64_t)b_len * 4,                   WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufALen     = makeBuf("aLen",       4,                                     WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufBLen     = makeBuf("bLen",       4,                                     WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufNumWg    = makeBuf("numWg",      4,                                     WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufDPI      = makeBuf("DPI",        (uint64_t)(2*(numWg+1))*4,             WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    WGPUBuffer bufState    = makeBuf("State",      (uint64_t)numWg * 4,                   WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufOutput   = makeBuf("Output",     (uint64_t)std::max(maxOutput, 1u) * 4, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    WGPUBuffer bufTotalCnt = makeBuf("TotalCount", 4,                                     WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst);

    wgpuQueueWriteBuffer(queue, bufA,     0, setA.data(), a_len * 4);
    wgpuQueueWriteBuffer(queue, bufB,     0, setB.data(), b_len * 4);
    wgpuQueueWriteBuffer(queue, bufALen,  0, &a_len, 4);
    wgpuQueueWriteBuffer(queue, bufBLen,  0, &b_len, 4);
    wgpuQueueWriteBuffer(queue, bufNumWg, 0, &numWg, 4);

    WGPUBindGroupEntry diagEntries[] = {
        bufEntry(0, bufA), bufEntry(1, bufB), bufEntry(2, bufDPI),
        bufEntry(3, bufALen), bufEntry(4, bufBLen), bufEntry(5, bufNumWg),
    };
    WGPUBindGroupDescriptor diagBGD{};
    diagBGD.label = wgpuStr("DPI BG");
    diagBGD.layout = diagBGL;
    diagBGD.entryCount = 6;
    diagBGD.entries = diagEntries;
    WGPUBindGroup diagBG = wgpuDeviceCreateBindGroup(device, &diagBGD);

    WGPUBindGroupEntry lbEntries[] = {
        bufEntry(0, bufA), bufEntry(1, bufB), bufEntry(2, bufDPI),
        bufEntry(3, bufState), bufEntry(4, bufOutput), bufEntry(5, bufTotalCnt),
        bufEntry(6, bufALen), bufEntry(7, bufBLen), bufEntry(8, bufNumWg),
    };
    WGPUBindGroupDescriptor lbBGD{};
    lbBGD.label = wgpuStr("LB BG");
    lbBGD.layout = lookbackBGL;
    lbBGD.entryCount = 9;
    lbBGD.entries = lbEntries;
    WGPUBindGroup lbBG = wgpuDeviceCreateBindGroup(device, &lbBGD);

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
    wgpuBufferRelease(bufA);
    wgpuBufferRelease(bufB);
    wgpuBufferRelease(bufALen);
    wgpuBufferRelease(bufBLen);
    wgpuBufferRelease(bufNumWg);
    wgpuBufferRelease(bufDPI);
    wgpuBufferRelease(bufState);
    wgpuBufferRelease(bufOutput);
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

    printf("=== Unified Set Operations (Balanced Path + Decoupled Lookback) ===\n");
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
    // NOTE: some headers return bool, some return status. handle both.
    auto retA = wgpuAdapterGetLimits(g_adapter, &adapterLimits);  // What is the upper limit you theoretically support?
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

    // Request the maximum the adapter says it supports.
    devDesc.requiredLimits = &adapterLimits;  // I hope the newly created device at least meets these limits.

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
    auto retD = wgpuDeviceGetLimits(g_device, &deviceLimits);  // What is the actual limit that ultimately takes effect for you?

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

    g_subgroupSize = 32; // you still assume 32; if you have an API to query subgroup size, plug it in.

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
        fprintf(stderr, "ERROR: shader directory not found. Run from common_cpp/not_by_key/ or build/Debug/\n");
        return 1;
    }
    char dpiPath[256], lbPath[256];
    snprintf(dpiPath, sizeof(dpiPath), "%s/balanced_path_biased.wgsl", shaderDir);
    snprintf(lbPath,  sizeof(lbPath),  "%s/set_availability_decoupled_lookback.wgsl", shaderDir);
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

    WGPUBindGroupLayoutEntry diagLayoutEntries[] = {
        bglEntry(0, WGPUBufferBindingType_ReadOnlyStorage),
        bglEntry(1, WGPUBufferBindingType_ReadOnlyStorage),
        bglEntry(2, WGPUBufferBindingType_Storage),
        bglEntry(3, WGPUBufferBindingType_Uniform),
        bglEntry(4, WGPUBufferBindingType_Uniform),
        bglEntry(5, WGPUBufferBindingType_Uniform),
    };
    WGPUBindGroupLayoutDescriptor diagBGLD{};
    diagBGLD.label = wgpuStr("DPI BGL");
    diagBGLD.entryCount = 6;
    diagBGLD.entries = diagLayoutEntries;
    WGPUBindGroupLayout diagBGL = wgpuDeviceCreateBindGroupLayout(g_device, &diagBGLD);

    WGPUBindGroupLayoutEntry lbLayoutEntries[] = {
        bglEntry(0, WGPUBufferBindingType_ReadOnlyStorage),
        bglEntry(1, WGPUBufferBindingType_ReadOnlyStorage),
        bglEntry(2, WGPUBufferBindingType_ReadOnlyStorage),
        bglEntry(3, WGPUBufferBindingType_Storage),
        bglEntry(4, WGPUBufferBindingType_Storage),
        bglEntry(5, WGPUBufferBindingType_Storage),
        bglEntry(6, WGPUBufferBindingType_Uniform),
        bglEntry(7, WGPUBufferBindingType_Uniform),
        bglEntry(8, WGPUBufferBindingType_Uniform),
    };
    WGPUBindGroupLayoutDescriptor lbBGLD{};
    lbBGLD.label = wgpuStr("Lookback BGL");
    lbBGLD.entryCount = 9;
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
        printf("  Set %s (OP_MODE=%u)\n", OP_NAMES[opMode], opMode);
        printf("========================================\n");

        std::string lbCode = setOpMode(lbCodeTemplate, opMode);
        WGPUShaderModule lbShaderModule = createShaderModule("Lookback Shader", lbCode);
        WGPUComputePipeline lbPipeline = makePipeline("Lookback Pipeline", lookbackBGL,
                                                      lbShaderModule, "decoupled_lookback_kernel");

        printf("Lookback pipeline created (OP_MODE=%u).\n", opMode);

        if (dataDir) {
            char pathA[256], pathB[256];
            snprintf(pathA, sizeof(pathA), "%s/A_1e2.bin", dataDir);
            snprintf(pathB, sizeof(pathB), "%s/B_1e2.bin", dataDir);

            std::ifstream fA(pathA), fB(pathB);
            if (fA.good() && fB.good()) {
                fA.close(); fB.close();
                auto A = loadBinaryData(pathA);
                auto B = loadBinaryData(pathB);

                printf("\nValidation (%zuM + %zuM):\n", A.size()/1000000, B.size()/1000000);
                bool ok = runValidation(g_device, queue,
                                        diagPipeline, diagBGL,
                                        lbPipeline, lookbackBGL,
                                        A, B, opMode);
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

                    uint32_t maxOut = getMaxOutputSize(opMode, (uint32_t)A.size(), (uint32_t)B.size());
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
                                               A, B, opMode, WARMUP, MEASURED);

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