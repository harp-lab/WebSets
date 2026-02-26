import { runUnified4PhaseVs2PhaseTest, runUnifiedSingleOpTest } from './balanced_path/common/not_by_key/test_unified_4phase_vs_2phase';
import { runUnifiedByKey4PhaseVs2PhaseTest, runUnifiedByKeySingleOpTest } from './balanced_path/common/by_key/test_unified_4phase_vs_2phase_by_key';

(async () => {
    if (navigator.gpu === undefined) {
        document.getElementById("webgpu-canvas").setAttribute("style", "display:none;");
        document.getElementById("no-webgpu").setAttribute("style", "display:block;");
        return;
    }

    // Get a GPU device to render with
    let adapter = await navigator.gpu.requestAdapter();
	const supportsTimestampQueries = adapter?.features.has('timestamp-query');
    const supportsSubgroups = adapter?.features.has('subgroups' as GPUFeatureName);
    const requiredFeatures: GPUFeatureName[] = [];
    if (supportsTimestampQueries) requiredFeatures.push('timestamp-query');
    if (supportsSubgroups) requiredFeatures.push('subgroups' as GPUFeatureName);
    let gpuDeviceDesc = {
        requiredLimits: {
            maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
            maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
            maxBufferSize: adapter.limits.maxBufferSize,
            maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
        },
        requiredFeatures,
    };
    let device = await adapter.requestDevice(gpuDeviceDesc);

    console.log("maxBufferSize =", adapter.limits.maxBufferSize);
    console.log("maxStorageBufferBindingSize =", adapter.limits.maxStorageBufferBindingSize);
    console.log("subgroups =", supportsSubgroups ? "supported" : "NOT supported");

    // =================================
    // UNIFIED: All 4 Ops 4-Phase vs 2-Phase
    // =================================
    // await runUnifiedSingleOpTest(device, 0);  // intersection only
    // await runUnifiedSingleOpTest(device, 1);  // difference only
    // await runUnifiedSingleOpTest(device, 2);  // union only
    // await runUnifiedSingleOpTest(device, 3);  // sym_difference only

    // =================================
    // UNIFIED BY KEY: All 4 Ops 4-Phase vs 2-Phase
    // =================================
    // await runUnifiedByKeySingleOpTest(device, 0);  // intersection by key only
    // await runUnifiedByKeySingleOpTest(device, 1);  // difference by key only
    // await runUnifiedByKeySingleOpTest(device, 2);  // union by key only
    await runUnifiedByKeySingleOpTest(device, 3);  // sym_difference by key only
})();

