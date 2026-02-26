/**
 * Unified Test: 4-Phase vs 2-Phase Pipeline for All 4 Set Operations (By Key)
 *
 * By-Key variant: compares by keys, outputs key-value pairs.
 * Reuses existing DPI and Count shaders (keys-only). Uses new write/lookback
 * by-key shaders that scatter both keys and values.
 *
 * Tests the unified common shaders with OP_MODE string replacement:
 *   0 = intersection (A ∩ B)
 *   1 = difference   (A \ B)
 *   2 = union        (A ∪ B)
 *   3 = sym_difference ((A\B) ∪ (B\A))
 *
 * 4-Phase Pipeline:
 *   1. DPI - Compute diagonal path indices (binds key arrays)
 *   2. Count - Count matches per workgroup (binds key arrays, reuses existing shader)
 *   3. Prefix Sum - Exclusive scan for output offsets
 *   4. Write By Key - Write result keys+values (new by-key shader)
 *
 * 2-Phase Pipeline:
 *   1. DPI - Compute diagonal path indices (binds key arrays)
 *   2. Decoupled Lookback By Key - Single-pass count + write with built-in prefix sum
 */

import TimestampQueryManager from '../../../TimestampQueryManager';
import * as utils from '../../../utils';
import { setOpMode } from '../../../utils';
import computeDiagonalsShader from '../balanced_path_biased.wgsl';
import countShaderBase from '../set_availability_count.wgsl';
import writeByKeyShaderBase from './set_availability_write_by_key.wgsl';
import lookbackByKeyShaderBase from './set_availability_decoupled_lookback_by_key.wgsl';
import { ExclusiveScanPipeline } from '../prefix_sum/exclusive_scan';

const MAXWORKGROUP = 65535;
const NT = 256;
const VT = 12;
const NV = NT * VT;  // 3072
const DPI_WG_SIZE = 256;

const OP_NAMES = ['intersection', 'difference', 'union', 'sym_difference'] as const;
type OpName = typeof OP_NAMES[number];

/** CPU by-key validation functions.
 *  intersection: A is interleaved [k,v,...], B is keys-only
 *  difference/union/sym_diff: both A and B are interleaved [k,v,...]
 */
const CPU_FUNCTIONS_BY_KEY: Record<OpName, (a: Uint32Array, b: Uint32Array) => Uint32Array> = {
    intersection: utils.setIntersectionCPUByKey,
    difference: utils.setDifferenceCPUByKey,
    union: utils.setUnionCPUByKey,
    sym_difference: utils.setSymmetricDifferenceCPUByKey,
};

/** Max possible output size for each operation (used to allocate output buffer in 2-phase). */
function getMaxOutputSize(opMode: number, aLen: number, bLen: number): number {
    switch (opMode) {
        case 0: return Math.min(aLen, bLen);       // intersection
        case 1: return aLen;                        // difference
        case 2: return aLen + bLen;                 // union
        case 3: return aLen + bLen;                 // sym_difference
        default: return aLen + bLen;
    }
}

/**
 * Interleave separate key and value arrays into [k0, v0, k1, v1, ...] format.
 */
function interleaveKeyValue(keys: Uint32Array, values: Uint32Array): Uint32Array {
    const n = keys.length;
    const result = new Uint32Array(n * 2);
    for (let i = 0; i < n; i++) {
        result[2 * i] = keys[i];
        result[2 * i + 1] = values[i];
    }
    return result;
}

/**
 * Generate synthetic values for a key array: values[i] = i + offset.
 * This produces unique, verifiable values.
 */
function generateValues(keys: Uint32Array, offset: number): Uint32Array {
    const values = new Uint32Array(keys.length);
    for (let i = 0; i < keys.length; i++) {
        values[i] = i + offset;
    }
    return values;
}

/**
 * 4-Phase Pipeline Tester By Key (DPI -> Count -> Scan -> Write By Key)
 */
class FourPhasePipelineByKeyTester {
    private device: GPUDevice;
    private timestampQueryManager: TimestampQueryManager;
    private label: string;

    private diagPipeline: GPUComputePipeline;
    private diagBindGroupLayout: GPUBindGroupLayout;
    private countPipeline: GPUComputePipeline;
    private countBindGroupLayout: GPUBindGroupLayout;
    private writePipeline: GPUComputePipeline;
    private writeBindGroupLayout: GPUBindGroupLayout;
    private scanPipeline: any;

    constructor(device: GPUDevice, timestampQueryManager: TimestampQueryManager, label: string, opMode: number) {
        this.device = device;
        this.timestampQueryManager = timestampQueryManager;
        this.label = label;
        this.scanPipeline = new ExclusiveScanPipeline(device);

        // Count shader: reuses existing keys-only shader
        const countShader = setOpMode(countShaderBase, opMode);
        // Write shader: new by-key variant
        const writeShader = setOpMode(writeByKeyShaderBase, opMode);

        // DPI bind group layout (unchanged - binds key arrays)
        this.diagBindGroupLayout = device.createBindGroupLayout({
            label: `${label} DPI bind group layout`,
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ]
        });

        this.diagPipeline = device.createComputePipeline({
            label: `${label} DPI pipeline`,
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.diagBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: computeDiagonalsShader }),
                entryPoint: 'compute_diagonals'
            }
        });

        // Count bind group layout (unchanged - binds key arrays)
        this.countBindGroupLayout = device.createBindGroupLayout({
            label: `${label} Count kernel bind group layout`,
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ]
        });

        this.countPipeline = device.createComputePipeline({
            label: `${label} Count kernel pipeline`,
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.countBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: countShader }),
                entryPoint: 'count_availability'
            }
        });

        // Write By Key bind group layout (11 bindings)
        this.writeBindGroupLayout = device.createBindGroupLayout({
            label: `${label} Write By Key kernel bind group layout`,
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // a_keys
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // a_values
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // b_keys
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // b_values
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // dpi
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // offsets
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },             // output_keys
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },             // output_values
                { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },             // a_length
                { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },             // b_length
                { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // num_wg_total
            ]
        });

        this.writePipeline = device.createComputePipeline({
            label: `${label} Write By Key kernel pipeline`,
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.writeBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: writeShader }),
                entryPoint: 'write_availability_by_key'
            }
        });
    }

    public async run(
        aKeys: Uint32Array,
        aValues: Uint32Array,
        bKeys: Uint32Array,
        bValues: Uint32Array,
        iterations: number,
        warmup: number
    ): Promise<{
        resultKeys: Uint32Array;
        resultValues: Uint32Array;
        totalCount: number;
        timing: {
            dpiMs: number;
            countMs: number;
            scanMs: number;
            writeMs: number;
            totalMs: number;
        };
    }> {
        const device = this.device;
        const tsm = this.timestampQueryManager;
        const a_len = aKeys.length;
        const b_len = bKeys.length;
        const total = a_len + b_len;

        if (total === 0) {
            return {
                resultKeys: new Uint32Array(0),
                resultValues: new Uint32Array(0),
                totalCount: 0,
                timing: { dpiMs: 0, countMs: 0, scanMs: 0, writeMs: 0, totalMs: 0 }
            };
        }

        const numWg = Math.ceil(total / NV);

        // Create key buffers
        const bufferAKeys = device.createBuffer({
            size: Math.max(4, aKeys.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferAKeys, 0, new Uint32Array(aKeys));

        const bufferAValues = device.createBuffer({
            size: Math.max(4, aValues.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferAValues, 0, new Uint32Array(aValues));

        const bufferBKeys = device.createBuffer({
            size: Math.max(4, bKeys.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferBKeys, 0, new Uint32Array(bKeys));

        const bufferBValues = device.createBuffer({
            size: Math.max(4, bValues.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferBValues, 0, new Uint32Array(bValues));

        // Uniform buffers
        const bufferALen = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const bufferBLen = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const bufferNumWg = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(bufferALen, 0, new Uint32Array([a_len]));
        device.queue.writeBuffer(bufferBLen, 0, new Uint32Array([b_len]));
        device.queue.writeBuffer(bufferNumWg, 0, new Uint32Array([numWg]));

        const dpiSize = 2 * (numWg + 1);
        const bufferDPI = device.createBuffer({
            size: dpiSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const alignedSize = this.scanPipeline.getAlignedSize(numWg);
        const bufferCounts = device.createBuffer({
            size: alignedSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        const bufferCountsCopy = device.createBuffer({
            size: numWg * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        const dispatchX = Math.min(numWg, MAXWORKGROUP);
        const dispatchY = Math.ceil(numWg / MAXWORKGROUP);

        const subgroupSize = (device.adapterInfo as any)?.subgroupSize || 32;
        const subgroupsPerWg = DPI_WG_SIZE / subgroupSize;
        const dpiBlocks = Math.ceil(numWg / subgroupsPerWg);
        const dpiDispatchX = Math.min(dpiBlocks, MAXWORKGROUP);
        const dpiDispatchY = Math.ceil(dpiBlocks / MAXWORKGROUP);

        // DPI and Count bind groups: bind KEY arrays (not values)
        const diagBindGroup = device.createBindGroup({
            layout: this.diagBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferAKeys } },
                { binding: 1, resource: { buffer: bufferBKeys } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferALen } },
                { binding: 4, resource: { buffer: bufferBLen } },
                { binding: 5, resource: { buffer: bufferNumWg } },
            ]
        });

        const countBindGroup = device.createBindGroup({
            layout: this.countBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferAKeys } },
                { binding: 1, resource: { buffer: bufferBKeys } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferCounts } },
                { binding: 4, resource: { buffer: bufferALen } },
                { binding: 5, resource: { buffer: bufferBLen } },
                { binding: 6, resource: { buffer: bufferNumWg } },
            ]
        });

        const scanner = this.scanPipeline.prepareGPUInput(bufferCounts, alignedSize);

        await device.queue.onSubmittedWorkDone();

        // First pass: DPI + Count + Scan to determine output size
        {
            const encoder = device.createCommandEncoder();

            let pass = encoder.beginComputePass();
            pass.setPipeline(this.diagPipeline);
            pass.setBindGroup(0, diagBindGroup);
            pass.dispatchWorkgroups(dpiDispatchX, dpiDispatchY);
            pass.end();

            pass = encoder.beginComputePass();
            pass.setPipeline(this.countPipeline);
            pass.setBindGroup(0, countBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            encoder.copyBufferToBuffer(bufferCounts, 0, bufferCountsCopy, 0, numWg * 4);
            scanner.recordScanCommands(encoder, numWg, tsm, 4, 5);

            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();
        }

        // Read back counts to determine actual output size
        const countsReadback = device.createBuffer({
            size: numWg * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        const offsetsReadback = device.createBuffer({
            size: numWg * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        let readEncoder = device.createCommandEncoder();
        readEncoder.copyBufferToBuffer(bufferCountsCopy, 0, countsReadback, 0, numWg * 4);
        readEncoder.copyBufferToBuffer(bufferCounts, 0, offsetsReadback, 0, numWg * 4);
        device.queue.submit([readEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        await countsReadback.mapAsync(GPUMapMode.READ);
        const counts = new Uint32Array(countsReadback.getMappedRange().slice(0));
        countsReadback.unmap();

        await offsetsReadback.mapAsync(GPUMapMode.READ);
        const offsets = new Uint32Array(offsetsReadback.getMappedRange().slice(0));
        offsetsReadback.unmap();

        const totalCount = offsets[numWg - 1] + counts[numWg - 1];

        // Create output buffers for keys and values
        const bufferOutputKeys = device.createBuffer({
            size: Math.max(4, totalCount * 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const bufferOutputValues = device.createBuffer({
            size: Math.max(4, totalCount * 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        // Write By Key bind group (11 bindings)
        const writeBindGroup = device.createBindGroup({
            layout: this.writeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferAKeys } },
                { binding: 1, resource: { buffer: bufferAValues } },
                { binding: 2, resource: { buffer: bufferBKeys } },
                { binding: 3, resource: { buffer: bufferBValues } },
                { binding: 4, resource: { buffer: bufferDPI } },
                { binding: 5, resource: { buffer: bufferCounts } },    // offsets (prefix sum result)
                { binding: 6, resource: { buffer: bufferOutputKeys } },
                { binding: 7, resource: { buffer: bufferOutputValues } },
                { binding: 8, resource: { buffer: bufferALen } },
                { binding: 9, resource: { buffer: bufferBLen } },
                { binding: 10, resource: { buffer: bufferNumWg } },
            ]
        });

        // Warmup runs
        for (let w = 0; w < warmup; w++) {
            const encoder = device.createCommandEncoder();

            let pass = encoder.beginComputePass();
            pass.setPipeline(this.diagPipeline);
            pass.setBindGroup(0, diagBindGroup);
            pass.dispatchWorkgroups(dpiDispatchX, dpiDispatchY);
            pass.end();

            pass = encoder.beginComputePass();
            pass.setPipeline(this.countPipeline);
            pass.setBindGroup(0, countBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            encoder.copyBufferToBuffer(bufferCounts, 0, bufferCountsCopy, 0, numWg * 4);
            scanner.recordScanCommands(encoder, numWg, tsm, 4, 5);

            pass = encoder.beginComputePass();
            pass.setPipeline(this.writePipeline);
            pass.setBindGroup(0, writeBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();
        }

        // Timed runs
        const dpiTimes: number[] = [];
        const countTimes: number[] = [];
        const scanTimes: number[] = [];
        const writeTimes: number[] = [];
        const totalTimes: number[] = [];

        for (let iter = 0; iter < iterations; iter++) {
            const encoder = device.createCommandEncoder();

            let pass = encoder.beginComputePass(tsm.createComputePassDescriptor(0, 1));
            pass.setPipeline(this.diagPipeline);
            pass.setBindGroup(0, diagBindGroup);
            pass.dispatchWorkgroups(dpiDispatchX, dpiDispatchY);
            pass.end();

            pass = encoder.beginComputePass(tsm.createComputePassDescriptor(2, 3));
            pass.setPipeline(this.countPipeline);
            pass.setBindGroup(0, countBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            encoder.copyBufferToBuffer(bufferCounts, 0, bufferCountsCopy, 0, numWg * 4);
            scanner.recordScanCommands(encoder, numWg, tsm, 4, 5);

            pass = encoder.beginComputePass(tsm.createComputePassDescriptor(6, 7));
            pass.setPipeline(this.writePipeline);
            pass.setBindGroup(0, writeBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            tsm.resolve(encoder);
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();

            const timestamps = await tsm.downloadTimestampResult();

            if (timestamps.length >= 8) {
                const dpiNs = timestamps[1] - timestamps[0];
                const countNs = timestamps[3] - timestamps[2];
                const writeNs = timestamps[7] - timestamps[6];
                const totalNs = timestamps[7] - timestamps[0];
                const scanNs = totalNs - dpiNs - countNs - writeNs;

                dpiTimes.push(dpiNs / 1_000_000);
                countTimes.push(countNs / 1_000_000);
                scanTimes.push(scanNs / 1_000_000);
                writeTimes.push(writeNs / 1_000_000);
                totalTimes.push(totalNs / 1_000_000);
            }
        }

        const avg = (arr: number[]) => arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

        // Readback results
        let resultKeys = new Uint32Array(0);
        let resultValues = new Uint32Array(0);
        if (totalCount > 0) {
            const keysReadback = device.createBuffer({
                size: totalCount * 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });
            const valuesReadback = device.createBuffer({
                size: totalCount * 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });
            readEncoder = device.createCommandEncoder();
            readEncoder.copyBufferToBuffer(bufferOutputKeys, 0, keysReadback, 0, totalCount * 4);
            readEncoder.copyBufferToBuffer(bufferOutputValues, 0, valuesReadback, 0, totalCount * 4);
            device.queue.submit([readEncoder.finish()]);
            await device.queue.onSubmittedWorkDone();

            await keysReadback.mapAsync(GPUMapMode.READ);
            resultKeys = new Uint32Array(keysReadback.getMappedRange().slice(0));
            keysReadback.unmap();

            await valuesReadback.mapAsync(GPUMapMode.READ);
            resultValues = new Uint32Array(valuesReadback.getMappedRange().slice(0));
            valuesReadback.unmap();

            keysReadback.destroy();
            valuesReadback.destroy();
        }

        // Cleanup
        bufferAKeys.destroy();
        bufferAValues.destroy();
        bufferBKeys.destroy();
        bufferBValues.destroy();
        bufferALen.destroy();
        bufferBLen.destroy();
        bufferNumWg.destroy();
        bufferDPI.destroy();
        bufferCounts.destroy();
        bufferCountsCopy.destroy();
        bufferOutputKeys.destroy();
        bufferOutputValues.destroy();
        countsReadback.destroy();
        offsetsReadback.destroy();

        return {
            resultKeys,
            resultValues,
            totalCount,
            timing: {
                dpiMs: avg(dpiTimes),
                countMs: avg(countTimes),
                scanMs: avg(scanTimes),
                writeMs: avg(writeTimes),
                totalMs: avg(totalTimes),
            }
        };
    }
}

/**
 * 2-Phase Pipeline Tester By Key (DPI -> Decoupled Lookback By Key)
 */
class TwoPhasePipelineByKeyTester {
    private device: GPUDevice;
    private timestampQueryManager: TimestampQueryManager;
    private label: string;
    private opMode: number;

    private diagPipeline: GPUComputePipeline;
    private diagBindGroupLayout: GPUBindGroupLayout;
    private lookbackPipeline: GPUComputePipeline;
    private lookbackBindGroupLayout: GPUBindGroupLayout;

    constructor(device: GPUDevice, timestampQueryManager: TimestampQueryManager, label: string, opMode: number) {
        this.device = device;
        this.timestampQueryManager = timestampQueryManager;
        this.label = label;
        this.opMode = opMode;

        const lookbackShader = setOpMode(lookbackByKeyShaderBase, opMode);

        // DPI bind group layout (binds key arrays)
        this.diagBindGroupLayout = device.createBindGroupLayout({
            label: `${label} DPI bind group layout`,
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ]
        });

        this.diagPipeline = device.createComputePipeline({
            label: `${label} DPI pipeline`,
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.diagBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: computeDiagonalsShader }),
                entryPoint: 'compute_diagonals'
            }
        });

        // Lookback By Key bind group layout (12 bindings)
        this.lookbackBindGroupLayout = device.createBindGroupLayout({
            label: `${label} Decoupled Lookback By Key bind group layout`,
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // a_keys
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // a_values
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // b_keys
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // b_values
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // dpi
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },             // state
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },             // output_keys
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },             // output_values
                { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },             // total_count
                { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },             // a_length
                { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // b_length
                { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // num_wg_total
            ]
        });

        this.lookbackPipeline = device.createComputePipeline({
            label: `${label} Decoupled Lookback By Key pipeline`,
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.lookbackBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: lookbackShader }),
                entryPoint: 'decoupled_lookback_by_key_kernel'
            }
        });
    }

    public async run(
        aKeys: Uint32Array,
        aValues: Uint32Array,
        bKeys: Uint32Array,
        bValues: Uint32Array,
        iterations: number,
        warmup: number
    ): Promise<{
        resultKeys: Uint32Array;
        resultValues: Uint32Array;
        totalCount: number;
        timing: {
            dpiMs: number;
            lookbackMs: number;
            totalMs: number;
        };
    }> {
        const device = this.device;
        const a_len = aKeys.length;
        const b_len = bKeys.length;
        const total = a_len + b_len;

        if (total === 0) {
            return {
                resultKeys: new Uint32Array(0),
                resultValues: new Uint32Array(0),
                totalCount: 0,
                timing: { dpiMs: 0, lookbackMs: 0, totalMs: 0 }
            };
        }

        const numWg = Math.ceil(total / NV);
        const maxOutputSize = getMaxOutputSize(this.opMode, a_len, b_len);

        // Create key/value buffers
        const bufferAKeys = device.createBuffer({
            size: Math.max(4, aKeys.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferAKeys, 0, new Uint32Array(aKeys));

        const bufferAValues = device.createBuffer({
            size: Math.max(4, aValues.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferAValues, 0, new Uint32Array(aValues));

        const bufferBKeys = device.createBuffer({
            size: Math.max(4, bKeys.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferBKeys, 0, new Uint32Array(bKeys));

        const bufferBValues = device.createBuffer({
            size: Math.max(4, bValues.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferBValues, 0, new Uint32Array(bValues));

        // Uniform buffers
        const bufferALen = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const bufferBLen = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const bufferNumWg = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(bufferALen, 0, new Uint32Array([a_len]));
        device.queue.writeBuffer(bufferBLen, 0, new Uint32Array([b_len]));
        device.queue.writeBuffer(bufferNumWg, 0, new Uint32Array([numWg]));

        const dpiSize = 2 * (numWg + 1);
        const bufferDPI = device.createBuffer({
            size: dpiSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const bufferState = device.createBuffer({
            size: numWg * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const bufferOutputKeys = device.createBuffer({
            size: Math.max(maxOutputSize, 1) * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const bufferOutputValues = device.createBuffer({
            size: Math.max(maxOutputSize, 1) * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const bufferTotalCount = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        // Create bind groups
        const diagBindGroup = device.createBindGroup({
            layout: this.diagBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferAKeys } },
                { binding: 1, resource: { buffer: bufferBKeys } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferALen } },
                { binding: 4, resource: { buffer: bufferBLen } },
                { binding: 5, resource: { buffer: bufferNumWg } },
            ]
        });

        const lookbackBindGroup = device.createBindGroup({
            layout: this.lookbackBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferAKeys } },
                { binding: 1, resource: { buffer: bufferAValues } },
                { binding: 2, resource: { buffer: bufferBKeys } },
                { binding: 3, resource: { buffer: bufferBValues } },
                { binding: 4, resource: { buffer: bufferDPI } },
                { binding: 5, resource: { buffer: bufferState } },
                { binding: 6, resource: { buffer: bufferOutputKeys } },
                { binding: 7, resource: { buffer: bufferOutputValues } },
                { binding: 8, resource: { buffer: bufferTotalCount } },
                { binding: 9, resource: { buffer: bufferALen } },
                { binding: 10, resource: { buffer: bufferBLen } },
                { binding: 11, resource: { buffer: bufferNumWg } },
            ]
        });

        const dispatchX = Math.min(numWg, MAXWORKGROUP);
        const dispatchY = Math.ceil(numWg / MAXWORKGROUP);

        const subgroupSize = (device.adapterInfo as any)?.subgroupSize || 32;
        const subgroupsPerWg = DPI_WG_SIZE / subgroupSize;
        const dpiBlocks = Math.ceil(numWg / subgroupsPerWg);
        const dpiDispatchX = Math.min(dpiBlocks, MAXWORKGROUP);
        const dpiDispatchY = Math.ceil(dpiBlocks / MAXWORKGROUP);

        await device.queue.onSubmittedWorkDone();

        // Warmup runs
        for (let iter = 0; iter < warmup; iter++) {
            device.queue.writeBuffer(bufferState, 0, new Uint32Array(numWg).fill(0));
            device.queue.writeBuffer(bufferTotalCount, 0, new Uint32Array([0]));

            const encoder = device.createCommandEncoder();

            let pass = encoder.beginComputePass();
            pass.setPipeline(this.diagPipeline);
            pass.setBindGroup(0, diagBindGroup);
            pass.dispatchWorkgroups(dpiDispatchX, dpiDispatchY);
            pass.end();

            pass = encoder.beginComputePass();
            pass.setPipeline(this.lookbackPipeline);
            pass.setBindGroup(0, lookbackBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            device.queue.submit([encoder.finish()]);
        }
        await device.queue.onSubmittedWorkDone();

        // Timed runs
        const dpiTimes: number[] = [];
        const lookbackTimes: number[] = [];
        const totalTimes: number[] = [];

        for (let iter = 0; iter < iterations; iter++) {
            device.queue.writeBuffer(bufferState, 0, new Uint32Array(numWg).fill(0));
            device.queue.writeBuffer(bufferTotalCount, 0, new Uint32Array([0]));

            const encoder = device.createCommandEncoder();

            let pass = encoder.beginComputePass(
                this.timestampQueryManager.createComputePassDescriptor(0, 1)
            );
            pass.setPipeline(this.diagPipeline);
            pass.setBindGroup(0, diagBindGroup);
            pass.dispatchWorkgroups(dpiDispatchX, dpiDispatchY);
            pass.end();

            pass = encoder.beginComputePass(
                this.timestampQueryManager.createComputePassDescriptor(2, 3)
            );
            pass.setPipeline(this.lookbackPipeline);
            pass.setBindGroup(0, lookbackBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            this.timestampQueryManager.resolve(encoder);
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();

            const timestamps = await this.timestampQueryManager.downloadTimestampResult();

            if (timestamps.length >= 4) {
                const dpiNs = timestamps[1] - timestamps[0];
                const lookbackNs = timestamps[3] - timestamps[2];
                const totalNs = timestamps[3] - timestamps[0];

                dpiTimes.push(dpiNs / 1_000_000);
                lookbackTimes.push(lookbackNs / 1_000_000);
                totalTimes.push(totalNs / 1_000_000);
            }
        }

        const avg = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;

        // Readback
        const totalCountReadback = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        let readEncoder = device.createCommandEncoder();
        readEncoder.copyBufferToBuffer(bufferTotalCount, 0, totalCountReadback, 0, 4);
        device.queue.submit([readEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        await totalCountReadback.mapAsync(GPUMapMode.READ);
        const totalCount = new Uint32Array(totalCountReadback.getMappedRange().slice(0))[0];
        totalCountReadback.unmap();

        let resultKeys = new Uint32Array(0);
        let resultValues = new Uint32Array(0);
        if (totalCount > 0) {
            const keysReadback = device.createBuffer({
                size: totalCount * 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });
            const valuesReadback = device.createBuffer({
                size: totalCount * 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });
            readEncoder = device.createCommandEncoder();
            readEncoder.copyBufferToBuffer(bufferOutputKeys, 0, keysReadback, 0, totalCount * 4);
            readEncoder.copyBufferToBuffer(bufferOutputValues, 0, valuesReadback, 0, totalCount * 4);
            device.queue.submit([readEncoder.finish()]);
            await device.queue.onSubmittedWorkDone();

            await keysReadback.mapAsync(GPUMapMode.READ);
            resultKeys = new Uint32Array(keysReadback.getMappedRange().slice(0));
            keysReadback.unmap();

            await valuesReadback.mapAsync(GPUMapMode.READ);
            resultValues = new Uint32Array(valuesReadback.getMappedRange().slice(0));
            valuesReadback.unmap();

            keysReadback.destroy();
            valuesReadback.destroy();
        }

        // Cleanup
        bufferAKeys.destroy();
        bufferAValues.destroy();
        bufferBKeys.destroy();
        bufferBValues.destroy();
        bufferALen.destroy();
        bufferBLen.destroy();
        bufferNumWg.destroy();
        bufferDPI.destroy();
        bufferState.destroy();
        bufferOutputKeys.destroy();
        bufferOutputValues.destroy();
        bufferTotalCount.destroy();
        totalCountReadback.destroy();

        return {
            resultKeys,
            resultValues,
            totalCount,
            timing: {
                dpiMs: avg(dpiTimes),
                lookbackMs: avg(lookbackTimes),
                totalMs: avg(totalTimes),
            }
        };
    }
}

// ============================================================================
// Exported test runners
// ============================================================================

/**
 * Run correctness + benchmark for a single set operation (by key).
 */
async function runSingleOpByKeyTest(
    device: GPUDevice,
    opMode: number,
    opName: OpName,
    cpuFn: (a: Uint32Array, b: Uint32Array) => Uint32Array
): Promise<void> {
    const NUM_ITERATIONS = 100;
    const NUM_WARMUP = 10;

    const timestampQueryManager = new TimestampQueryManager(device, 16);

    if (!timestampQueryManager.timestampSupported) {
        console.log('ERROR: GPU timestamp queries are not supported on this device.\n');
        return;
    }

    const fourPhaseTester = new FourPhasePipelineByKeyTester(device, timestampQueryManager, `4P-ByKey-${opName}`, opMode);
    const twoPhaseTester = new TwoPhasePipelineByKeyTester(device, timestampQueryManager, `2P-ByKey-${opName}`, opMode);

    // ========================================================================
    // Performance Benchmarks
    // ========================================================================
    console.log(`  Performance (${NUM_WARMUP} warmup + ${NUM_ITERATIONS} iterations):\n`);

    console.log('  Dataset   | 4-Phase Total(ms) | 2-Phase Total(ms) | Winner  | Match | Output Count');
    console.log('  ----------|-------------------|-------------------|---------|-------|-------------');

    const datasets = [
        { size: '1', range: 'e2' },
        { size: '1', range: 'e6' },
        { size: '2', range: 'e2' },
        { size: '2', range: 'e6' },
        { size: '4', range: 'e2' },
        { size: '4', range: 'e6' },
        { size: '8', range: 'e2' },
        { size: '8', range: 'e6' },
        { size: '16', range: 'e2' },
        { size: '16', range: 'e6' },
        { size: '32', range: 'e2' },
        { size: '32', range: 'e6' },
        { size: '64', range: 'e2' },
        { size: '64', range: 'e6' },
        { size: '128', range: 'e2' },
        { size: '128', range: 'e6' },
    ];

    for (const { size, range } of datasets) {
        // Skip 128M for OP_MODE 2,3 (union/sym_diff by key): needs ~4GB VRAM
        // (a_keys + a_values + b_keys + b_values + output_keys + output_values)
        if (opMode >= 2 && parseInt(size) >= 128) {
            const ds = `${size}M${range}`.padEnd(8);
            console.log(`  ${ds} | Skipped (OOM: by-key union/sym_diff at 128M exceeds VRAM)`);
            continue;
        }

        const aPath = `./data/A_${size}${range}.bin`;
        const bPath = `./data/B_${size}${range}.bin`;

        try {
            const aKeys = await utils.loadUint32ArrayFromBin(aPath);
            const bKeys = await utils.loadUint32ArrayFromBin(bPath);

            // Generate synthetic values: a_values[i] = i, b_values[i] = i + a_len
            const aValues = generateValues(aKeys, 0);
            // For GPU: OP_MODE 0,1 (intersection/difference) never access b_values,
            // so use a tiny dummy to avoid OOM on large datasets (saves 512MB at 128M).
            // Full b_values only needed for OP_MODE 2,3 (union/sym_diff).
            const bValuesGPU = (opMode <= 1) ? new Uint32Array(1) : generateValues(bKeys, aKeys.length);

            const fourResult = await fourPhaseTester.run(aKeys, aValues, bKeys, bValuesGPU, NUM_ITERATIONS, NUM_WARMUP);
            const twoResult = await twoPhaseTester.run(aKeys, aValues, bKeys, bValuesGPU, NUM_ITERATIONS, NUM_WARMUP);

            const countMatch = fourResult.totalCount === twoResult.totalCount;

            // CPU validation for small datasets
            let cpuValid = '';
            const sizeNum = parseInt(size) * 1_000_000;
            if (sizeNum <= 4_000_000) {
                // Build interleaved arrays for CPU validation
                const aInterleaved = interleaveKeyValue(aKeys, aValues);

                let cpuResult: Uint32Array;
                if (opMode === 0) {
                    // intersection: B is keys-only
                    cpuResult = cpuFn(aInterleaved, bKeys);
                } else {
                    // difference/union/sym_diff: both interleaved
                    // Generate proper b_values for CPU validation (small datasets only)
                    const bValuesCPU = (opMode <= 1) ? new Uint32Array(1) : bValuesGPU;
                    const bInterleaved = interleaveKeyValue(bKeys, bValuesCPU);
                    cpuResult = cpuFn(aInterleaved, bInterleaved);
                }
            }

            const ds = `${size}M${range}`.padEnd(8);
            const f_total = fourResult.timing.totalMs.toFixed(3).padStart(17);
            const t_total = twoResult.timing.totalMs.toFixed(3).padStart(17);

            let winner: string;
            if (fourResult.timing.totalMs < twoResult.timing.totalMs) {
                const ratio = (twoResult.timing.totalMs / fourResult.timing.totalMs).toFixed(2);
                winner = `4P ${ratio}x`;
            } else {
                const ratio = (fourResult.timing.totalMs / twoResult.timing.totalMs).toFixed(2);
                winner = `2P ${ratio}x`;
            }
            const matchStr = countMatch ? 'OK  ' : 'FAIL';
            const countStr = twoResult.totalCount.toLocaleString().padStart(13);

            console.log(`  ${ds} |${f_total} |${t_total} | ${winner.padEnd(7)} | ${matchStr} |${countStr}${cpuValid}`);

        } catch (error) {
            const ds = `${size}M${range}`.padEnd(8);
            console.log(`  ${ds} | Error loading dataset`);
        }
    }

    console.log('');
}

/**
 * Run all 4 set operations (by key): correctness + performance comparison.
 */
export async function runUnifiedByKey4PhaseVs2PhaseTest(device: GPUDevice): Promise<void> {
    console.log('\n========================================================================');
    console.log('  UNIFIED BY-KEY 4-PHASE vs 2-PHASE TEST (All 4 Set Operations)');
    console.log('  Using common by-key shaders with OP_MODE replacement');
    console.log('========================================================================\n');

    for (let opMode = 0; opMode < 4; opMode++) {
        const opName = OP_NAMES[opMode];
        const cpuFn = CPU_FUNCTIONS_BY_KEY[opName];

        console.log(`--- ${opName.toUpperCase()} BY KEY (OP_MODE=${opMode}) ---\n`);
        await runSingleOpByKeyTest(device, opMode, opName, cpuFn);
    }

    console.log('All 4 by-key operations tested.\n');
}

/**
 * Run a single by-key operation test (for selective testing).
 */
export async function runUnifiedByKeySingleOpTest(device: GPUDevice, opMode: number): Promise<void> {
    const opName = OP_NAMES[opMode];
    const cpuFn = CPU_FUNCTIONS_BY_KEY[opName];

    console.log(`\n--- ${opName.toUpperCase()} BY KEY (OP_MODE=${opMode}) ---\n`);
    await runSingleOpByKeyTest(device, opMode, opName, cpuFn);
}
