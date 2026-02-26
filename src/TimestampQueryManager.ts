export default class TimestampQueryManager {
  timestampSupported: boolean;
  #timestampQuerySet: GPUQuerySet;
  #timestampBuffer: GPUBuffer;
  #timestampMapBuffer: GPUBuffer;

  constructor(device: GPUDevice, queryCount: number) {
    this.timestampSupported = device.features.has('timestamp-query');
    if (!this.timestampSupported) return;

    const timestampByteSize = 8;
    this.#timestampQuerySet = device.createQuerySet({
      type: 'timestamp',
      count: queryCount,
    });

    this.#timestampBuffer = device.createBuffer({
      size: this.#timestampQuerySet.count * timestampByteSize,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.QUERY_RESOLVE,
    });

    this.#timestampMapBuffer = device.createBuffer({
      size: this.#timestampBuffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  createComputePassDescriptor(beginIndex: number, endIndex: number): GPUComputePassDescriptor {
    if (!this.timestampSupported) return {};
    return {
      timestampWrites: {
        querySet: this.#timestampQuerySet,
        beginningOfPassWriteIndex: beginIndex,
        endOfPassWriteIndex: endIndex,
      }
    };
  }

  /**
   * Create compute pass descriptor with only begin timestamp.
   * Useful for measuring the start of multi-pass operations like Scan.
   */
  createComputePassDescriptorBeginOnly(beginIndex: number): GPUComputePassDescriptor {
    if (!this.timestampSupported) return {};
    return {
      timestampWrites: {
        querySet: this.#timestampQuerySet,
        beginningOfPassWriteIndex: beginIndex,
      }
    };
  }

  /**
   * Create compute pass descriptor with only end timestamp.
   * Useful for measuring the end of multi-pass operations like Scan.
   */
  createComputePassDescriptorEndOnly(endIndex: number): GPUComputePassDescriptor {
    if (!this.timestampSupported) return {};
    return {
      timestampWrites: {
        querySet: this.#timestampQuerySet,
        endOfPassWriteIndex: endIndex,
      }
    };
  }

  resolve(commandEncoder: GPUCommandEncoder) {
    if (!this.timestampSupported) return;

    commandEncoder.resolveQuerySet(
      this.#timestampQuerySet,
      0,
      this.#timestampQuerySet.count,
      this.#timestampBuffer,
      0
    );

    commandEncoder.copyBufferToBuffer(
      this.#timestampBuffer,
      0,
      this.#timestampMapBuffer,
      0,
      this.#timestampBuffer.size
    );
  }

  async downloadTimestampResult(): Promise<number[]> {
    if (!this.timestampSupported) return [];
    await this.#timestampMapBuffer.mapAsync(GPUMapMode.READ);
    const rawData = this.#timestampMapBuffer.getMappedRange();
    const timestamps = new BigUint64Array(rawData);
    const result = Array.from(timestamps, t => Number(t));
    this.#timestampMapBuffer.unmap();
    return result;
  }
}