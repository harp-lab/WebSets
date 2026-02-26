export function alignTo(val: number, align: number): number;

export class ExclusiveScanPipeline {
    constructor(device: GPUDevice);
    device: GPUDevice;
    workGroupSize: number;
    maxScanSize: number;
    clearBuf: GPUBuffer;
    scanBlocksLayout: GPUBindGroupLayout;
    scanBlockResultsLayout: GPUBindGroupLayout;
    scanBlocksPipeline: GPUComputePipeline;
    scanBlockResultsPipeline: GPUComputePipeline;
    addBlockSumsPipeline: GPUComputePipeline;
    getAlignedSize(size: number): number;
    prepareInput(cpuArray: Uint32Array): any;
    prepareGPUInput(gpuBuffer: GPUBuffer, alignedSize: number): any;
}
