export async function readSize(device: GPUDevice, bufferSize: GPUBuffer): Promise<number> {
    const gpuReadBufferSize = device.createBuffer({
        label: 'Read size',
        size: Uint32Array.BYTES_PER_ELEMENT, 
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const commandEncoderSize = device.createCommandEncoder();
    commandEncoderSize.copyBufferToBuffer(bufferSize, 0, gpuReadBufferSize, 0, 4);
    device.queue.submit([commandEncoderSize.finish()]);

    // Map the buffer and read its value
    await gpuReadBufferSize.mapAsync(GPUMapMode.READ);
    const arrayBufferSize = gpuReadBufferSize.getMappedRange();
    const sizeValue = new Uint32Array(arrayBufferSize)[0];
    gpuReadBufferSize.unmap();
    return sizeValue;
}

export async function loadUint32ArrayFromBin(url: string): Promise<Uint32Array> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}: ${res.status} ${res.statusText}`);
  }

  const buffer = await res.arrayBuffer();
  const view = new DataView(buffer);

  const length = Number(view.getBigUint64(0, true)); // little-endian = true

  const data = new Uint32Array(buffer, 8, length);

  return data;
}

/**
 * Replace the compile-time OP_MODE constant in a unified WGSL shader.
 *   0 = intersection, 1 = difference, 2 = union, 3 = sym_difference
 */
export function setOpMode(shader: string, mode: number): string {
    return shader.replace('const OP_MODE: u32 = 0u;', `const OP_MODE: u32 = ${mode}u;`);
}

export function setIntersectionCPU(setA: Uint32Array, setB: Uint32Array): Uint32Array {
    const result: number[] = [];

    let i = 0;
    let j = 0;

    while (i < setA.length && j < setB.length) {
        const a = setA[i];
        const b = setB[j];
        if (a < b) {
            i++;
        } else if (b < a) {
            j++;
        } else {
            result.push(a);
            i++;
            j++;
        }
    }

    console.log(result.length);

    return new Uint32Array(result);
}

export function setIntersectionCPUUnsorted(
  setA: Uint32Array,
  setB: Uint32Array
): Uint32Array {
  const a = new Uint32Array(setA);
  const b = new Uint32Array(setB);

  a.sort();
  b.sort();

  const result: number[] = [];
  let i = 0;
  let j = 0;

  while (i < a.length && j < b.length) {
    const va = a[i];
    const vb = b[j];

    if (va < vb) {
      i++;
    } else if (va > vb) {
      j++;
    } else {
      result.push(va);
      i++;
      j++;
    }
  }

  return new Uint32Array(result);
}

export function setIntersectionCPUByKey(
  setA: Uint32Array,  // flattened [k0, v0, k1, v1, ...]
  setB: Uint32Array   // keys only [b0, b1, ...]
): Uint32Array {
  if (setA.length % 2 !== 0) {
    throw new Error(
      "setA must contain (key, value) pairs, so its length must be even."
    );
  }

  const numPairsA = setA.length / 2;
  const lenB = setB.length;

  let iA = 0;  // index over pairs in A
  let iB = 0;  // index over keys in B

  const result: number[] = [];

  while (iA < numPairsA && iB < lenB) {
    const keyA = setA[2 * iA];
    const keyB = setB[iB];

    if (keyA < keyB) {
      iA++;
    } else if (keyA > keyB) {
      iB++;
    } else {
      const key = keyA;

      const startA = iA;
      while (iA < numPairsA && setA[2 * iA] === key) {
        iA++;
      }
      const mA = iA - startA;

      const startB = iB;
      while (iB < lenB && setB[iB] === key) {
        iB++;
      }
      const nB = iB - startB;

      const common = mA < nB ? mA : nB;

      for (let t = 0; t < common; ++t) {
        const pairIndex = startA + t;
        const outKey = setA[2 * pairIndex];
        const outVal = setA[2 * pairIndex + 1];
        result.push(outKey, outVal);
      }
    }
  }

  return new Uint32Array(result);
}

export function setIntersectionCPUByKeyUnsorted(
  setA: Uint32Array,  // flattened [k0, v0, k1, v1, ...] (unsorted)
  setB: Uint32Array   // keys only [b0, b1, ...] (unsorted)
): Uint32Array {
  if (setA.length % 2 !== 0) {
    throw new Error(
      "setA must contain (key, value) pairs, so its length must be even."
    );
  }

  const numPairsA = setA.length / 2;
  const lenB = setB.length;

  // ---------- 1) 先对 A 的 key 排序（保持 value 为 payload） ----------
  // 使用索引数组保证排序稳定性（同 key 时按原始顺序）
  const indicesA = new Array<number>(numPairsA);
  for (let i = 0; i < numPairsA; ++i) {
    indicesA[i] = i;
  }

  indicesA.sort((i, j) => {
    const keyAi = setA[2 * i];
    const keyAj = setA[2 * j];
    if (keyAi < keyAj) return -1;
    if (keyAi > keyAj) return 1;
    // key 相等时按原始索引保证稳定
    return i - j;
  });

  const sortedA = new Uint32Array(setA.length);
  {
    let pos = 0;
    for (const idx of indicesA) {
      sortedA[pos++] = setA[2 * idx];       // key
      sortedA[pos++] = setA[2 * idx + 1];   // value
    }
  }

  // ---------- 2) 对 B 的 key 排序 ----------
  const bArray = Array.from(setB);  // 先转成普通数组方便 sort
  bArray.sort((a, b) => a - b);
  const sortedB = new Uint32Array(bArray);

  // ---------- 3) 在排好序的 A 和 B 上做按 key 的 intersection ----------
  let iA = 0;  // index over pairs in sortedA
  let iB = 0;  // index over keys in sortedB

  const result: number[] = [];

  while (iA < numPairsA && iB < lenB) {
    const keyA = sortedA[2 * iA];
    const keyB = sortedB[iB];

    if (keyA < keyB) {
      iA++;
    } else if (keyA > keyB) {
      iB++;
    } else {
      const key = keyA;

      // 统计 A 中当前 key 连续段长度
      const startA = iA;
      while (iA < numPairsA && sortedA[2 * iA] === key) {
        iA++;
      }
      const mA = iA - startA;

      // 统计 B 中当前 key 连续段长度
      const startB = iB;
      while (iB < lenB && sortedB[iB] === key) {
        iB++;
      }
      const nB = iB - startB;

      // 交集 multiplicity = min(mA, nB)
      const common = mA < nB ? mA : nB;

      // 按 A 里的顺序输出对应数量的 (key, value) pair
      for (let t = 0; t < common; ++t) {
        const pairIndex = startA + t;
        const outKey = sortedA[2 * pairIndex];
        const outVal = sortedA[2 * pairIndex + 1];
        result.push(outKey, outVal);
      }
    }
  }

  return new Uint32Array(result);
}



export function setDifferenceCPU(setA: Uint32Array, setB: Uint32Array): Uint32Array {
    const countB = new Map<number, number>();
    for (const v of setB) {
        countB.set(v, (countB.get(v) ?? 0) + 1);
    }

    const result: number[] = [];
    for (const v of setA) {
        const c = countB.get(v) ?? 0;
        if (c > 0) {
            countB.set(v, c - 1);
        } else {
            result.push(v);
        }
    }

    return new Uint32Array(result);
}


export function setDifferenceCPUByKey(
  setA: Uint32Array,
  setB: Uint32Array
): Uint32Array {
  if (setA.length % 2 !== 0 || setB.length % 2 !== 0) {
    throw new Error(
      "setA and setB must contain (key, value) pairs (length must be even)."
    );
  }

  const nA = setA.length / 2;
  const nB = setB.length / 2;

  const result: number[] = [];

  let iA = 0;
  let iB = 0;

  while (iA < nA && iB < nB) {
    const keyA = setA[2 * iA];
    const keyB = setB[2 * iB];

    if (keyA < keyB) {
      const runStartA = iA;
      while (iA < nA && setA[2 * iA] === keyA) {
        iA++;
      }
      const runEndA = iA;
      const m = runEndA - runStartA;

      for (let idx = runStartA; idx < runEndA; ++idx) {
        const k = setA[2 * idx];
        const v = setA[2 * idx + 1];
        result.push(k, v);
      }
    } else if (keyB < keyA) {
      const runKeyB = keyB;
      while (iB < nB && setB[2 * iB] === runKeyB) {
        iB++;
      }
    } else {
      const runKey = keyA;

      const runStartA = iA;
      while (iA < nA && setA[2 * iA] === runKey) {
        iA++;
      }
      const runEndA = iA;
      const m = runEndA - runStartA;

      const runStartB = iB;
      while (iB < nB && setB[2 * iB] === runKey) {
        iB++;
      }
      const runEndB = iB;
      const n = runEndB - runStartB;

      const keep = Math.max(m - n, 0);
      const firstKeptOffset = m - keep;

      for (let offset = firstKeptOffset; offset < m; ++offset) {
        const idxA = runStartA + offset;
        const k = setA[2 * idxA];
        const v = setA[2 * idxA + 1];
        result.push(k, v);
      }
    }
  }

  while (iA < nA) {
    const keyA = setA[2 * iA];
    const runStartA = iA;
    while (iA < nA && setA[2 * iA] === keyA) {
      iA++;
    }
    const runEndA = iA;

    for (let idx = runStartA; idx < runEndA; ++idx) {
      const k = setA[2 * idx];
      const v = setA[2 * idx + 1];
      result.push(k, v);
    }
  }

  return new Uint32Array(result);
}


export function setUnionCPU(setA: Uint32Array, setB: Uint32Array): Uint32Array {
    const result: number[] = [];
    let i = 0, j = 0;
    
    while (i < setA.length && j < setB.length) {
        if (setA[i] < setB[j]) {
            result.push(setA[i]);
            i++;
        } else if (setA[i] === setB[j]) {
            result.push(setA[i]);
            i++;
            j++;
        } else {
            result.push(setB[j]);
            j++;
        }
    }
    
    while (i < setA.length) {
        result.push(setA[i]);
        i++;
    }
    while (j < setB.length) {
        result.push(setB[j]);
        j++;
    }
    
    return new Uint32Array(result);
}

export function setUnionCPUByKey(
  setA: Uint32Array,
  setB: Uint32Array
): Uint32Array {
  if (setA.length % 2 !== 0 || setB.length % 2 !== 0) {
    throw new Error("setA and setB must contain (key, value) pairs (length must be even).");
  }

  const lenA = setA.length / 2; 
  const lenB = setB.length / 2; 

  let i = 0; // index in pairs for A
  let j = 0; // index in pairs for B

  const result: number[] = [];

  while (i < lenA && j < lenB) {
    const keyA = setA[2 * i];
    const keyB = setB[2 * j];

    if (keyA < keyB) {
      const key = keyA;
      const startA = i;
      while (i < lenA && setA[2 * i] === key) {
        i++;
      }
      for (let p = startA; p < i; ++p) {
        result.push(setA[2 * p], setA[2 * p + 1]);
      }
    } else if (keyB < keyA) {
      const key = keyB;
      const startB = j;
      while (j < lenB && setB[2 * j] === key) {
        j++;
      }
      for (let p = startB; p < j; ++p) {
        result.push(setB[2 * p], setB[2 * p + 1]);
      }
    } else {
      const key = keyA;

      const startA = i;
      while (i < lenA && setA[2 * i] === key) {
        i++;
      }
      const endA = i;
      const m = endA - startA;

      const startB = j;
      while (j < lenB && setB[2 * j] === key) {
        j++;
      }
      const endB = j;
      const n = endB - startB;

      for (let p = startA; p < endA; ++p) {
        result.push(setA[2 * p], setA[2 * p + 1]);
      }

      if (n > m) {
        const extra = n - m;
        const beginExtra = endB - extra; 
        for (let p = beginExtra; p < endB; ++p) {
          result.push(setB[2 * p], setB[2 * p + 1]);
        }
      }
    }
  }

  while (i < lenA) {
    const key = setA[2 * i];
    const startA = i;
    while (i < lenA && setA[2 * i] === key) {
      i++;
    }
    for (let p = startA; p < i; ++p) {
      result.push(setA[2 * p], setA[2 * p + 1]);
    }
  }

  while (j < lenB) {
    const key = setB[2 * j];
    const startB = j;
    while (j < lenB && setB[2 * j] === key) {
      j++;
    }
    for (let p = startB; p < j; ++p) {
      result.push(setB[2 * p], setB[2 * p + 1]);
    }
  }

  return new Uint32Array(result);
}


export function setSymmetricDifferenceCPU(setA: Uint32Array, setB: Uint32Array): Uint32Array {
    const bSet = new Set(setB);
    const result: number[] = [];

    for (const elem of setA) {
        if (!bSet.has(elem)) {
            result.push(elem);
        }
    }

    for (const elem of setB) {
        if (!setA.includes(elem)) {
            result.push(elem);
        }
    }

    return new Uint32Array(result.sort((a, b) => a - b));
}

export function setSymmetricDifferenceCPUByKey(
  setA: Uint32Array,
  setB: Uint32Array
): Uint32Array {
  if ((setA.length & 1) !== 0 || (setB.length & 1) !== 0) {
    throw new Error("setA and setB must contain (key, value) pairs (length must be even).");
  }

  const result: number[] = [];
  let iA = 0;
  let iB = 0;

  while (iA < setA.length && iB < setB.length) {
    const keyA = setA[iA];
    const keyB = setB[iB];

    if (keyA < keyB) {
      const key = keyA;
      while (iA < setA.length && setA[iA] === key) {
        result.push(setA[iA], setA[iA + 1]);
        iA += 2;
      }
    } else if (keyB < keyA) {
      const key = keyB;
      while (iB < setB.length && setB[iB] === key) {
        result.push(setB[iB], setB[iB + 1]);
        iB += 2;
      }
    } else {
      const key = keyA;

      const startA = iA;
      const startB = iB;

      let m = 0;
      while (iA < setA.length && setA[iA] === key) {
        iA += 2;
        ++m;
      }

      let n = 0;
      while (iB < setB.length && setB[iB] === key) {
        iB += 2;
        ++n;
      }

      if (m > n) {
        const keep = m - n;
        let startIdx = startA + 2 * n;
        const endIdx = startA + 2 * m;
        for (let idx = startIdx; idx < endIdx; idx += 2) {
          result.push(setA[idx], setA[idx + 1]);
        }
      } else if (n > m) {
        const keep = n - m;
        let startIdx = startB + 2 * m; 
        const endIdx = startB + 2 * n;
        for (let idx = startIdx; idx < endIdx; idx += 2) {
          result.push(setB[idx], setB[idx + 1]);
        }
      }
    }
  }

  while (iA < setA.length) {
    const key = setA[iA];
    while (iA < setA.length && setA[iA] === key) {
      result.push(setA[iA], setA[iA + 1]);
      iA += 2;
    }
  }

  while (iB < setB.length) {
    const key = setB[iB];
    while (iB < setB.length && setB[iB] === key) {
      result.push(setB[iB], setB[iB + 1]);
      iB += 2;
    }
  }

  return new Uint32Array(result);
}



export function compareResults(gpuResult: Uint32Array, cpuResult: Uint32Array): boolean {
    if (gpuResult.length !== cpuResult.length) {
        console.error(`Length mismatch: GPU=${gpuResult.length}, CPU=${cpuResult.length}`);
        return false;
    }

    for (let i = 0; i < gpuResult.length; i++) {
        if (gpuResult[i] !== cpuResult[i]) {
            console.error(`Mismatch at index ${i}: GPU=${gpuResult[i]}, CPU=${cpuResult[i]}`);
            return false;
        }
    }
    // console.log(`Result validation PASSED: ${gpuResult.length} elements match`);
    return true;
}

export function getNeighbors(
      rowPtr: Uint32Array,
      neighbors: Uint32Array,
      u: number
  ): Uint32Array {
      const start = rowPtr[u];
      const end   = rowPtr[u + 1];
      return neighbors.slice(start, end);
  }


function intersectSorted(a: Uint32Array, b: Uint32Array): number {
    let i = 0, j = 0, count = 0;
    while (i < a.length && j < b.length) {
        if (a[i] < b[j]) i++;
        else if (a[i] > b[j]) j++;
        else {
        count++;
        i++;
        j++;
        }
    }
    return count;
}

export function countTrianglesCPU(
  numVertices: number,
  rowPtr: Uint32Array,
  neighbors: Uint32Array
): number {
  let total = 0;

  for (let u = 0; u < numVertices; u++) {
      const Nu = getNeighbors(rowPtr, neighbors, u);

      // 遍历每条 forward edge (u, v)
      for (let e = rowPtr[u]; e < rowPtr[u + 1]; e++) {
      const v = neighbors[e];
      const Nv = getNeighbors(rowPtr, neighbors, v);

      // 三角形数量 += |N⁺(u) ∩ N⁺(v)|
      const c = intersectSorted(Nu, Nv);
      total += c;
      }
  }

  return total;
}