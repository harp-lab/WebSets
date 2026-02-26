/*
    This file implements a gpu version of radix sort for 32-bit keys only.

    A good introduction to general purpose radix sort can be found here:
    http://www.codercorner.com/RadixSortRevisited.htm

    The gpu radix sort implemented here is a re-implementation of the Vulkan radix sort found in the fuchsia repos:
    https://fuchsia.googlesource.com/fuchsia/+/refs/heads/main/src/graphics/lib/compute/radix_sort/

    Modified to sort keys only (no payload/values) for use with atomic intersection output.

    All shaders can be found in radix_sort.wgsl
*/
import radix_sort from "./radix_sort.wgsl";
import TimestampQueryManager from '../../../TimestampQueryManager';

// IMPORTANT: the following constants have to be synced with the numbers in radix_sort.wgsl

/// workgroup size of histogram shader
const HISTOGRAM_WG_SIZE = 256;

/// one thread operates on 2 prefixes at the same time
const PREFIX_WG_SIZE = 1 << 7;

/// scatter compute shader work group size
const SCATTER_WG_SIZE = 1 << 8;

/// we sort 8 bits per pass
const RS_RADIX_LOG2 = 8;

/// 256 entries into the radix table
const RS_RADIX_SIZE = 1 << RS_RADIX_LOG2;

/// number of bytes our keys and values have
const RS_KEYVAL_SIZE = 32 / RS_RADIX_LOG2;

/// TODO describe me
const RS_HISTOGRAM_BLOCK_ROWS = 15;

/// DO NOT CHANGE, shader assume this!!!
const RS_SCATTER_BLOCK_ROWS = RS_HISTOGRAM_BLOCK_ROWS;

/// number of elements scattered by one work group
const SCATTER_BLOCK_KVS = HISTOGRAM_WG_SIZE * RS_SCATTER_BLOCK_ROWS;

/// number of elements scattered by one work group
const HISTO_BLOCK_KVS = HISTOGRAM_WG_SIZE * RS_HISTOGRAM_BLOCK_ROWS;

/// bytes per value
/// currently only 4 byte values are allowed
const BYTES_PER_PAYLOAD_ELEM = 4;

/// number of passed used for sorting
/// we sort 8 bits per pass so 4 passes are required for a 32 bit value
const NUM_PASSES = BYTES_PER_PAYLOAD_ELEM;

export class GPUSorter {
    zeroPipeline: GPUComputePipeline;
    histogramPipeline: GPUComputePipeline;
    prefixPipeline: GPUComputePipeline;
    scatterEvenPipeline: GPUComputePipeline;
    scatterOddPipeline: GPUComputePipeline;
    device: GPUDevice;
    bindGroupLayout: GPUBindGroupLayout;
    timestampQueryManager: TimestampQueryManager;

    constructor(device: GPUDevice, subgroupSize: number, timestampQueryManager: TimestampQueryManager) {
        this.device = device;
        this.timestampQueryManager = timestampQueryManager;
        let histogram_sg_size = subgroupSize;
        let rs_sweep_0_size = Math.floor(RS_RADIX_SIZE / histogram_sg_size);
        let rs_sweep_1_size = Math.floor(rs_sweep_0_size / histogram_sg_size);
        let rs_sweep_2_size = Math.floor(rs_sweep_1_size / histogram_sg_size);
        let rs_sweep_size = rs_sweep_0_size + rs_sweep_1_size + rs_sweep_2_size;
        let _rs_smem_phase_1 = RS_RADIX_SIZE + RS_RADIX_SIZE + rs_sweep_size;
        let rs_smem_phase_2 = RS_RADIX_SIZE + RS_SCATTER_BLOCK_ROWS * SCATTER_WG_SIZE;
        // rs_smem_phase_2 will always be larger, so always use phase2
        let rs_mem_dwords = rs_smem_phase_2;
        let rs_mem_sweep_0_offset = 0;
        let rs_mem_sweep_1_offset = rs_mem_sweep_0_offset + rs_sweep_0_size;
        let rs_mem_sweep_2_offset = rs_mem_sweep_1_offset + rs_sweep_1_size;
        // console.log(rs_mem_sweep_2_offset);

        this.bindGroupLayout = this.device.createBindGroupLayout({
            label: "radix sort bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" }
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" }
                },
            ]
        });
        const pipelineLayout = this.device.createPipelineLayout({
            label: "radix sort pipeline layout",
            bindGroupLayouts: [this.bindGroupLayout],
        });
        let shader_code = `
        const histogram_sg_size: u32 = ${histogram_sg_size}u;
        const histogram_wg_size: u32 = ${HISTOGRAM_WG_SIZE}u;
        const rs_radix_log2: u32 = ${RS_RADIX_LOG2}u;
        const rs_radix_size: u32 = ${RS_RADIX_SIZE}u;
        const rs_keyval_size: u32 = ${RS_KEYVAL_SIZE}u;
        const rs_histogram_block_rows: u32 = ${RS_HISTOGRAM_BLOCK_ROWS}u;
        const rs_scatter_block_rows: u32 = ${RS_SCATTER_BLOCK_ROWS}u;
        const rs_mem_dwords: u32 = ${rs_mem_dwords}u;
        const rs_mem_sweep_0_offset: u32 = ${rs_mem_sweep_0_offset}u;
        const rs_mem_sweep_1_offset: u32 = ${rs_mem_sweep_1_offset}u;
        const rs_mem_sweep_2_offset: u32 = ${rs_mem_sweep_2_offset}u;
        ${radix_sort}
        `;
        shader_code = shader_code
            .replace(/{histogram_wg_size}/g, HISTOGRAM_WG_SIZE.toString())
            .replace(/{prefix_wg_size}/g, PREFIX_WG_SIZE.toString())
            .replace(/{scatter_wg_size}/g, SCATTER_WG_SIZE.toString());
        const shader = this.device.createShaderModule({
            label: "Radix sort shader",
            code: shader_code,
        });

        this.zeroPipeline = this.device.createComputePipeline({
            label: "zero_histograms",
            layout: pipelineLayout,
            compute: {
                module: shader,
                entryPoint: "zero_histograms"
            }
        });
        this.histogramPipeline = this.device.createComputePipeline({
            label: "calculate_histogram",
            layout: pipelineLayout,
            compute: {
                module: shader,
                entryPoint: "calculate_histogram"
            }
        });
        this.prefixPipeline = this.device.createComputePipeline({
            label: "prefix_histogram",
            layout: pipelineLayout,
            compute: {
                module: shader,
                entryPoint: "prefix_histogram"
            }
        });
        this.scatterEvenPipeline = this.device.createComputePipeline({
            label: "scatter_even",
            layout: pipelineLayout,
            compute: {
                module: shader,
                entryPoint: "scatter_even"
            }
        });
        this.scatterOddPipeline = this.device.createComputePipeline({
            label: "scatter_odd",
            layout: pipelineLayout,
            compute: {
                module: shader,
                entryPoint: "scatter_odd"
            }
        });
    }

    public createKeyBuffers(length: number): [GPUBuffer, GPUBuffer] {
        // add padding so that our buffer size is a multiple of keys_per_workgroup
        let count_ru_histo = this.keysBufferSize(length) * RS_KEYVAL_SIZE;

        // creating the two needed buffers for sorting
        let keys = this.device.createBuffer({
            label: "radix sort keys buffer",
            size: count_ru_histo * BYTES_PER_PAYLOAD_ELEM,
            usage: GPUBufferUsage.STORAGE
                | GPUBufferUsage.COPY_DST
                | GPUBufferUsage.COPY_SRC,
        });

        // auxiliary buffer for keys
        let keys_aux = this.device.createBuffer({
            label: "radix sort keys auxiliary buffer",
            size: count_ru_histo * BYTES_PER_PAYLOAD_ELEM,
            usage: GPUBufferUsage.STORAGE,
        });

        return [keys, keys_aux];
    }

    // calculates and allocates a buffer that is sufficient for holding all needed information for
    // sorting. This includes the histograms and the temporary scatter buffer
    // @return: tuple containing [internal memory buffer (should be bound at shader binding 1, count_ru_histo (padded size needed for the keyval buffer)]
    public createInternalMemBuffer(length: number): GPUBuffer {
        // currently only a few different key bits are supported, maybe has to be extended

        // The "internal" memory map looks like this:
        //   +---------------------------------+ <-- 0
        //   | histograms[keyval_size]         |
        //   +---------------------------------+ <-- keyval_size                           * histo_size
        //   | partitions[scatter_blocks_ru-1] |
        //   +---------------------------------+ <-- (keyval_size + scatter_blocks_ru - 1) * histo_size
        //   | workgroup_ids[keyval_size]      |
        //   +---------------------------------+ <-- (keyval_size + scatter_blocks_ru - 1) * histo_size + workgroup_ids_size

        let s_b_ru = this.scatterBlocksRu(length);

        let histo_size = RS_RADIX_SIZE * 4;

        let internal_size = (RS_KEYVAL_SIZE + s_b_ru) * histo_size; // +1 safety

        let buffer = this.device.createBuffer({
            label: "Internal radix sort buffer",
            size: internal_size,
            usage: GPUBufferUsage.STORAGE,
        });
        return buffer;
    }

    public createSortBuffers(length: number): SortBuffers {
        const [keysA, keysB] = this.createKeyBuffers(length);
        const internalMemBuffer = this.createInternalMemBuffer(length);

        let uniform_infos = {
            num_keys: length,
            padded_size: this.keysBufferSize(length),
            even_pass: 0,
            odd_pass: 0,
        };
        let uniformBuffer = this.device.createBuffer({
            label: "radix sort uniform buffer",
            size: 4 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        })
        const uniformData = new Uint32Array([
            uniform_infos.num_keys,
            uniform_infos.padded_size,
            uniform_infos.even_pass,
            uniform_infos.odd_pass
        ]);
        this.device.queue.writeBuffer(uniformBuffer, 0, uniformData);

        const bindGroup = this.device.createBindGroup({
            label: "radix sort bind group",
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: internalMemBuffer } },
                { binding: 2, resource: { buffer: keysA } },
                { binding: 3, resource: { buffer: keysB } },
            ]
        });

        return new SortBuffers(keysA, keysB, internalMemBuffer, uniformBuffer, bindGroup, length);
    }
    
    public recordCalculateHistogram(commandEncoder: GPUCommandEncoder, bindGroup: GPUBindGroup, length: number, passDescriptor: GPUComputePassDescriptor = {}) {
        const histBlocksRu = this.histoBlocksRu(length);

        const passEncoder = commandEncoder.beginComputePass({ label: "zeroing histogram", ...passDescriptor });
        passEncoder.setPipeline(this.zeroPipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(histBlocksRu, 1, 1);
        passEncoder.end();

        const passEncoder2 = commandEncoder.beginComputePass({ label: "calculate histogram" });
        passEncoder2.setPipeline(this.histogramPipeline);
        passEncoder2.setBindGroup(0, bindGroup);
        passEncoder2.dispatchWorkgroups(histBlocksRu, 1, 1);
        passEncoder2.end();
    }

    public recordPrefixHistogram(commandEncoder: GPUCommandEncoder, bindGroup: GPUBindGroup) {
        const passEncoder = commandEncoder.beginComputePass({ label: "prefix histogram" });
        passEncoder.setPipeline(this.prefixPipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(NUM_PASSES, 1, 1);
        passEncoder.end();
    }

    public recordScatterKeys(commandEncoder: GPUCommandEncoder, bindGroup: GPUBindGroup, length: number, passDescriptor: GPUComputePassDescriptor = {}) {
        const scatterBlocksRu = this.scatterBlocksRu(length);

        const passEncoder = commandEncoder.beginComputePass({ label: "Scatter keyvals", ...passDescriptor });
        passEncoder.setBindGroup(0, bindGroup);

        passEncoder.setPipeline(this.scatterEvenPipeline);
        passEncoder.dispatchWorkgroups(scatterBlocksRu, 1, 1);

        passEncoder.setPipeline(this.scatterOddPipeline);
        passEncoder.dispatchWorkgroups(scatterBlocksRu, 1, 1);

        passEncoder.setPipeline(this.scatterEvenPipeline);
        passEncoder.dispatchWorkgroups(scatterBlocksRu, 1, 1);

        passEncoder.setPipeline(this.scatterOddPipeline);
        passEncoder.dispatchWorkgroups(scatterBlocksRu, 1, 1);

        passEncoder.end();
    }

    /// Writes sort commands to command encoder.
    /// If sort_first_n is not none one the first n elements are sorted
    /// otherwise everything is sorted.
    ///
    /// **IMPORTANT**: if less than the whole buffer is sorted the rest of the keys buffer will be be corrupted
    /// @param useTimestamps - If true, records GPU timestamps for the sort operation (default: false)
    public async sort(queue: GPUQueue, sortBuffers: SortBuffers, sortFirstN?: number, useTimestamps: boolean = false) {
        const numElements = sortFirstN ?? sortBuffers.length;

        // Update state buffer
        queue.writeBuffer(sortBuffers.uniformBuffer, 0, new Uint32Array([numElements]));

        const commandEncoder = this.device.createCommandEncoder();

        if (useTimestamps && this.timestampQueryManager.timestampSupported) {
            // 只在第一个 pass 记录开始时间，最后一个 pass 记录结束时间
            const beginPassDescriptor = this.timestampQueryManager.createComputePassDescriptorBeginOnly(0);
            const endPassDescriptor = this.timestampQueryManager.createComputePassDescriptorEndOnly(1);

            this.recordCalculateHistogram(commandEncoder, sortBuffers.bindGroup, numElements, beginPassDescriptor);
            this.recordPrefixHistogram(commandEncoder, sortBuffers.bindGroup);
            this.recordScatterKeys(commandEncoder, sortBuffers.bindGroup, numElements, endPassDescriptor);
            this.timestampQueryManager.resolve(commandEncoder);
        } else {
            // No timestamps - just record the compute passes
            this.recordCalculateHistogram(commandEncoder, sortBuffers.bindGroup, numElements);
            this.recordPrefixHistogram(commandEncoder, sortBuffers.bindGroup);
            this.recordScatterKeys(commandEncoder, sortBuffers.bindGroup, numElements);
        }

        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
    }

    public scatterBlocksRu(n: number): number {
        return Math.ceil(n / SCATTER_BLOCK_KVS);
    }

    public histoBlocksRu(n: number): number {
        return Math.ceil((this.scatterBlocksRu(n) * SCATTER_BLOCK_KVS) / HISTO_BLOCK_KVS);
    }

    public keysBufferSize(n: number): number {
        return this.histoBlocksRu(n) * HISTO_BLOCK_KVS;
    }
}

export class SortBuffers {
    constructor(
        public keysA: GPUBuffer,
        public keysB: GPUBuffer,
        public internalMemBuffer: GPUBuffer,
        public uniformBuffer: GPUBuffer,
        public bindGroup: GPUBindGroup,
        public length: number
    ) {}

    get keys(): GPUBuffer {
        return this.keysA;
    }

    keysValidSize(): number {
        return this.length * RS_KEYVAL_SIZE;
    }

    destroy() {
        this.keysA.destroy();
        this.keysB.destroy();
        this.internalMemBuffer.destroy();
        this.uniformBuffer.destroy();
    }
}