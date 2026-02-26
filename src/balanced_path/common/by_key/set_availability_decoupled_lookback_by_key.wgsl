// ============================================================================
// Unified Set Availability - Decoupled Lookback By Key (Sentinel + VT=12)
//
// By-Key variant: compares by keys, outputs key-value pairs.
// Single-pass approach using Decoupled Lookback for prefix sum computation.
// Combines Count + Scan + Write in one kernel.
//
// Supports all 4 set operations via compile-time OP_MODE constant:
//   0 = intersection (A ∩ B)
//   1 = difference   (A \ B)
//   2 = union        (A ∪ B)
//   3 = sym_difference ((A\B) ∪ (B\A))
//
// Sentinel optimization applied:
// - Add sentinel values at boundaries to eliminate bounds checking in binary search
// - NEG_INF (0) at start, POS_INF (0xFFFFFFFF) at end of each array section
// - All data indices shifted by +1 to accommodate leading sentinel
//
// Flow:
// 1. Read partition boundaries from DPI
// 2. Load A keys and A values into shared memory with sentinels
//    (B values read from global memory to limit shared memory usage)
// 3. Each thread runs Local BalancedPath to find its starting position
// 4. Each thread runs SerialSetOperationByKey to get results
// 5. Workgroup exclusive scan to compute local offsets and total
// 6. Thread 0 performs Decoupled Lookback to get global offset
// 7. All threads scatter result keys and values to output
// ============================================================================

// ============================================================================
// Operation Mode (override via string replacement in TypeScript)
//   0 = intersection, 1 = difference, 2 = union, 3 = sym_difference
// ============================================================================
const OP_MODE: u32 = 0u;

// ============================================================================
// Bindings (12 total for by-key variant)
// ============================================================================
@group(0) @binding(0) var<storage, read> a_keys: array<u32>;
@group(0) @binding(1) var<storage, read> a_values: array<u32>;
@group(0) @binding(2) var<storage, read> b_keys: array<u32>;
@group(0) @binding(3) var<storage, read> b_values: array<u32>;
@group(0) @binding(4) var<storage, read> dpi: array<u32>;
@group(0) @binding(5) var<storage, read_write> state: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> output_keys: array<u32>;
@group(0) @binding(7) var<storage, read_write> output_values: array<u32>;
@group(0) @binding(8) var<storage, read_write> total_count: array<atomic<u32>, 1>;
@group(0) @binding(9) var<uniform> a_length: u32;
@group(0) @binding(10) var<uniform> b_length: u32;
@group(0) @binding(11) var<uniform> num_wg_total: u32;

// ============================================================================
// Constants
// ============================================================================
const NT: u32 = 256u;           // Threads per workgroup
const VT: u32 = 12u;            // Values per thread
const NV: u32 = NT * VT;        // Total elements per workgroup = 3072

const STAR_MASK: u32 = 0x80000000u;
const INDEX_MASK: u32 = 0x7FFFFFFFu;
const MAX_DISPATCH_X: u32 = 65535u;

// Decoupled Lookback state flags
const STATUS_NOT_READY: u32 = 0u;
const STATUS_PARTIAL: u32 = 1u;
const STATUS_INCLUSIVE: u32 = 2u;

// State encoding: bits 31-30 = flag, bits 29-0 = value
const STATUS_SHIFT: u32 = 30u;
const VALUE_MASK: u32 = 0x3FFFFFFFu;

// Sentinel values
const POS_INF: u32 = 0xFFFFFFFFu;
const NEG_INF: u32 = 0u;

// ============================================================================
// Shared Memory Layout with Sentinels
// ============================================================================
// keys_shared: A keys + B keys with sentinel slots (3090 u32)
// a_values_shared: A values with sentinel slots (3090 u32)
//   - B values are NOT loaded to shared memory (read from global memory)
//   - This limits shared memory usage (~25KB total, fits on user's GPU)
//
// Total size: NV + VT + 2 + 4 = 3090
var<workgroup> keys_shared: array<u32, 3090>;
var<workgroup> a_values_shared: array<u32, 3090>;

// Workgroup-level variables
var<workgroup> wg_a0: u32;
var<workgroup> wg_a1: u32;
var<workgroup> wg_b0: u32;
var<workgroup> wg_b1: u32;
var<workgroup> wg_a_count: u32;
var<workgroup> wg_b_count: u32;
var<workgroup> wg_b_start: u32;
var<workgroup> wg_extended: bool;
var<workgroup> wg_bit0: u32;

// For workgroup-level scan
var<workgroup> shared_scan: array<u32, NT>;
var<workgroup> wg_local_total: u32;
var<workgroup> wg_exclusive_prefix: u32;

// ============================================================================
// State Pack/Unpack Functions
// ============================================================================
fn pack_state(flag: u32, value: u32) -> u32 {
    return (flag << STATUS_SHIFT) | (value & VALUE_MASK);
}

fn unpack_flag(packed: u32) -> u32 {
    return packed >> STATUS_SHIFT;
}

fn unpack_value(packed: u32) -> u32 {
    return packed & VALUE_MASK;
}

// ============================================================================
// Helper: Convert 2D workgroup_id to 1D index
// ============================================================================
fn get_workgroup_index(wg_id: vec3<u32>) -> u32 {
    return wg_id.x + wg_id.y * MAX_DISPATCH_X;
}

// ============================================================================
// DeviceLoad2ToShared: Cooperative loading with sentinels (By-Key variant)
// Loads A keys + A values into shared memory. B keys only (no B values).
// Data is offset by +1 to make room for leading sentinel.
// ============================================================================
fn device_load_2_to_shared(
    tid: u32,
    a_global_offset: u32,
    a_load_count: u32,
    b_global_offset: u32,
    b_load_count: u32,
    b_shared_start: u32
) {
    // Write sentinels (only thread 0)
    if (tid == 0u) {
        keys_shared[0] = NEG_INF;                        // A leading sentinel
        keys_shared[a_load_count + 1u] = POS_INF;        // A trailing sentinel
        keys_shared[b_shared_start] = NEG_INF;           // B leading sentinel
        keys_shared[b_shared_start + b_load_count + 1u] = POS_INF;  // B trailing sentinel
    }

    // Load A keys and A values: unrolled 12 iterations, with +1 offset for sentinel
    if (tid < a_load_count) {
        keys_shared[tid + 1u] = a_keys[a_global_offset + tid];
        a_values_shared[tid + 1u] = a_values[a_global_offset + tid];
    }
    if (tid + 256u < a_load_count) {
        keys_shared[tid + 257u] = a_keys[a_global_offset + tid + 256u];
        a_values_shared[tid + 257u] = a_values[a_global_offset + tid + 256u];
    }
    if (tid + 512u < a_load_count) {
        keys_shared[tid + 513u] = a_keys[a_global_offset + tid + 512u];
        a_values_shared[tid + 513u] = a_values[a_global_offset + tid + 512u];
    }
    if (tid + 768u < a_load_count) {
        keys_shared[tid + 769u] = a_keys[a_global_offset + tid + 768u];
        a_values_shared[tid + 769u] = a_values[a_global_offset + tid + 768u];
    }
    if (tid + 1024u < a_load_count) {
        keys_shared[tid + 1025u] = a_keys[a_global_offset + tid + 1024u];
        a_values_shared[tid + 1025u] = a_values[a_global_offset + tid + 1024u];
    }
    if (tid + 1280u < a_load_count) {
        keys_shared[tid + 1281u] = a_keys[a_global_offset + tid + 1280u];
        a_values_shared[tid + 1281u] = a_values[a_global_offset + tid + 1280u];
    }
    if (tid + 1536u < a_load_count) {
        keys_shared[tid + 1537u] = a_keys[a_global_offset + tid + 1536u];
        a_values_shared[tid + 1537u] = a_values[a_global_offset + tid + 1536u];
    }
    if (tid + 1792u < a_load_count) {
        keys_shared[tid + 1793u] = a_keys[a_global_offset + tid + 1792u];
        a_values_shared[tid + 1793u] = a_values[a_global_offset + tid + 1792u];
    }
    if (tid + 2048u < a_load_count) {
        keys_shared[tid + 2049u] = a_keys[a_global_offset + tid + 2048u];
        a_values_shared[tid + 2049u] = a_values[a_global_offset + tid + 2048u];
    }
    if (tid + 2304u < a_load_count) {
        keys_shared[tid + 2305u] = a_keys[a_global_offset + tid + 2304u];
        a_values_shared[tid + 2305u] = a_values[a_global_offset + tid + 2304u];
    }
    if (tid + 2560u < a_load_count) {
        keys_shared[tid + 2561u] = a_keys[a_global_offset + tid + 2560u];
        a_values_shared[tid + 2561u] = a_values[a_global_offset + tid + 2560u];
    }
    if (tid + 2816u < a_load_count) {
        keys_shared[tid + 2817u] = a_keys[a_global_offset + tid + 2816u];
        a_values_shared[tid + 2817u] = a_values[a_global_offset + tid + 2816u];
    }

    // Load B keys only: unrolled 12 iterations, with +1 offset for sentinel
    // (B values are read from global memory in serial_set_operation_by_key)
    if (tid < b_load_count) {
        keys_shared[b_shared_start + tid + 1u] = b_keys[b_global_offset + tid];
    }
    if (tid + 256u < b_load_count) {
        keys_shared[b_shared_start + tid + 257u] = b_keys[b_global_offset + tid + 256u];
    }
    if (tid + 512u < b_load_count) {
        keys_shared[b_shared_start + tid + 513u] = b_keys[b_global_offset + tid + 512u];
    }
    if (tid + 768u < b_load_count) {
        keys_shared[b_shared_start + tid + 769u] = b_keys[b_global_offset + tid + 768u];
    }
    if (tid + 1024u < b_load_count) {
        keys_shared[b_shared_start + tid + 1025u] = b_keys[b_global_offset + tid + 1024u];
    }
    if (tid + 1280u < b_load_count) {
        keys_shared[b_shared_start + tid + 1281u] = b_keys[b_global_offset + tid + 1280u];
    }
    if (tid + 1536u < b_load_count) {
        keys_shared[b_shared_start + tid + 1537u] = b_keys[b_global_offset + tid + 1536u];
    }
    if (tid + 1792u < b_load_count) {
        keys_shared[b_shared_start + tid + 1793u] = b_keys[b_global_offset + tid + 1792u];
    }
    if (tid + 2048u < b_load_count) {
        keys_shared[b_shared_start + tid + 2049u] = b_keys[b_global_offset + tid + 2048u];
    }
    if (tid + 2304u < b_load_count) {
        keys_shared[b_shared_start + tid + 2305u] = b_keys[b_global_offset + tid + 2304u];
    }
    if (tid + 2560u < b_load_count) {
        keys_shared[b_shared_start + tid + 2561u] = b_keys[b_global_offset + tid + 2560u];
    }
    if (tid + 2816u < b_load_count) {
        keys_shared[b_shared_start + tid + 2817u] = b_keys[b_global_offset + tid + 2816u];
    }
}

// ============================================================================
// Local Balanced Path - Simplified Binary Search with Sentinel Offset
//
// Since VT=12 < 16, get_local_biased_levels(VT) always returns 0.
// Simplified to standard binary search for better performance.
// Physical indices are +1 from logical indices.
// ============================================================================

// Standard lower_bound for A in shared memory (with sentinel offset)
fn lower_bound_local_a(end_exclusive: u32, key: u32) -> u32 {
    var lo: u32 = 0u;
    var hi: u32 = end_exclusive;

    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (keys_shared[mid + 1u] < key) { lo = mid + 1u; } else { hi = mid; }
    }

    return lo;
}

// Standard lower_bound for B in shared memory (with sentinel offset)
fn lower_bound_local_b(b_start: u32, end_exclusive: u32, key: u32) -> u32 {
    var lo: u32 = 0u;
    var hi: u32 = end_exclusive;

    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (keys_shared[b_start + mid + 1u] < key) { lo = mid + 1u; } else { hi = mid; }
    }

    return lo;
}

// Upper_bound for B in shared memory (with sentinel offset)
fn upper_bound_local_b(b_start: u32, range_begin: u32, range_end: u32, key: u32) -> u32 {
    var lo: u32 = range_begin;
    var hi: u32 = range_end;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (keys_shared[b_start + mid + 1u] <= key) { lo = mid + 1u; } else { hi = mid; }
    }
    return lo;
}

// Basic MergePath search in shared memory (with sentinel offset)
fn merge_path_local(a_count: u32, b_start: u32, b_count: u32, diag: u32) -> u32 {
    var lo: u32 = select(0u, diag - b_count, diag > b_count);
    var hi: u32 = min(diag, a_count);

    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        let a_key = keys_shared[mid + 1u];
        let b_key = keys_shared[b_start + diag - mid];
        if (a_key <= b_key) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }

    return lo;
}

// Complete Balanced Path with adjustment (with sentinel offset)
fn balanced_path_local(
    a_count: u32,
    b_start: u32,
    b_count: u32,
    diag: i32
) -> vec2<u32> {
    let diag_u = u32(max(0, diag));

    if (diag_u == 0u) {
        return vec2<u32>(0u, 0u);
    }
    if (diag_u >= a_count + b_count) {
        return vec2<u32>(a_count, 0u);
    }

    let p = merge_path_local(a_count, b_start, b_count, diag_u);

    var a_index = p;
    var b_index = diag_u - p;
    var star: u32 = 0u;

    if (b_index < b_count) {
        let x = keys_shared[b_start + b_index + 1u];
        let a_start = lower_bound_local_a(a_index, x);
        let b_start_run = lower_bound_local_b(b_start, b_index, x);
        let a_run = a_index - a_start;
        let b_run = b_index - b_start_run;
        let x_count = a_run + b_run;
        var b_advance = max(x_count >> 1u, x_count - a_run);
        var b_end_hint = min(b_count, b_start_run + b_advance + 1u);
        b_end_hint = max(b_end_hint, min(b_count, b_index + 1u));
        let b_run_end = upper_bound_local_b(b_start, b_index, b_end_hint, x);
        let actual_b_run = b_run_end - b_start_run;
        b_advance = min(b_advance, actual_b_run);
        let a_advance = x_count - b_advance;

        let round_up = (a_advance == b_advance + 1u) && (b_advance < actual_b_run);
        star = select(0u, 1u, round_up);

        a_index = a_start + a_advance;
    }

    return vec2<u32>(a_index, star);
}

// ============================================================================
// Serial Set Operation By Key - with sentinel offset
//
// By-Key variant: tracks values alongside keys.
// A values are loaded from shared memory (a_values_shared).
// B values are loaded from global memory (b_values) when needed.
//
// Commit conditions are IDENTICAL to the non-by-key version.
// ============================================================================
fn serial_set_operation_by_key(
    a_begin: u32,
    a_end: u32,
    b_begin: u32,
    b_end: u32,
    b_start_base: u32,
    b_global_offset: u32,
    star: u32,
    b_adjust: u32,
    extended: bool,
    result_keys: ptr<function, array<u32, 12>>,
    result_values: ptr<function, array<u32, 12>>
) -> u32 {
    var commit: u32 = 0u;
    var a_idx = a_begin;
    var b_idx = b_begin;

    var end_diag = i32(a_begin) + i32(b_begin) + i32(VT) - i32(star) - i32(b_adjust);
    if (!extended) {
        end_diag = min(end_diag, i32(a_end) + i32(b_end));
    }

    let min_iterations = VT / 2u;

    // Pre-fetch first keys (physical indices have +1 offset)
    var a_key = keys_shared[a_idx + 1u];
    var b_key = keys_shared[b_idx + 1u];
    // Pre-fetch first A value from shared memory
    var a_val = a_values_shared[a_idx + 1u];

    for (var i: u32 = 0u; i < VT; i++) {
        var test: bool;
        if (extended) {
            test = (i < min_iterations) || (i32(a_idx) + i32(b_idx) < end_diag);
        } else {
            test = (i32(a_idx) + i32(b_idx) < end_diag);
        }

        if (!test) {
            break;
        }

        let pA = a_key < b_key;
        let pB = b_key < a_key;

        // ============ OP_MODE: result value (by key) ============
        if (OP_MODE <= 1u) {
            // intersection, difference: always output A's key+value
            (*result_keys)[i] = a_key;
            (*result_values)[i] = a_val;
        } else {
            // union, sym_difference: output B when B < A
            (*result_keys)[i] = select(a_key, b_key, pB);
            if (pB) {
                // B value from global memory
                (*result_values)[i] = b_values[b_global_offset + (b_idx - b_start_base)];
            } else {
                (*result_values)[i] = a_val;
            }
        }

        // ============ OP_MODE: commit condition (IDENTICAL to non-by-key) ============
        var do_emit: bool;
        if (OP_MODE == 0u) {
            do_emit = (pA == pB);           // intersection: both equal
        } else if (OP_MODE == 1u) {
            do_emit = pA;                    // difference: A < B
        } else if (OP_MODE == 2u) {
            do_emit = true;                  // union: always emit
        } else {
            do_emit = (pA != pB);            // sym_difference: exactly one
        }
        if (do_emit) {
            commit |= (1u << i);
        }

        // Branchless pointer advancement using select()
        a_idx += select(0u, 1u, !pB);
        b_idx += select(0u, 1u, !pA);

        // Pre-fetch next keys (with +1 physical offset)
        a_key = select(a_key, keys_shared[a_idx + 1u], !pB);
        b_key = select(b_key, keys_shared[b_idx + 1u], !pA);
        // Pre-fetch next A value
        a_val = select(a_val, a_values_shared[a_idx + 1u], !pB);
    }

    return commit;
}

// countOneBits: using WGSL built-in function (no custom implementation needed)

// ============================================================================
// Workgroup Exclusive Scan with Total
// ============================================================================
fn workgroup_exclusive_scan_with_total(tid: u32, value: u32, total_ptr: ptr<function, u32>) -> u32 {
    shared_scan[tid] = value;
    workgroupBarrier();

    // Inclusive scan (Hillis-Steele style)
    for (var offset: u32 = 1u; offset < NT; offset *= 2u) {
        var temp: u32 = 0u;
        if (tid >= offset) {
            temp = shared_scan[tid - offset];
        }
        workgroupBarrier();
        shared_scan[tid] += temp;
        workgroupBarrier();
    }

    // Get total (last element's inclusive value)
    *total_ptr = shared_scan[NT - 1u];

    // Convert to exclusive scan
    if (tid == 0u) {
        return 0u;
    } else {
        return shared_scan[tid - 1u];
    }
}

// ============================================================================
// Decoupled Lookback: Compute global exclusive prefix for this workgroup
// ============================================================================
fn decoupled_lookback(wg_id: u32, local_total: u32) -> u32 {
    var exclusive_prefix: u32 = 0u;

    if (wg_id == 0u) {
        atomicStore(&state[0u], pack_state(STATUS_INCLUSIVE, local_total));
        return 0u;
    }

    atomicStore(&state[wg_id], pack_state(STATUS_PARTIAL, local_total));

    var lookback_id: i32 = i32(wg_id) - 1;
    var running_sum: u32 = 0u;

    while (lookback_id >= 0) {
        var predecessor_state: u32;
        loop {
            predecessor_state = atomicLoad(&state[u32(lookback_id)]);
            let flag = unpack_flag(predecessor_state);
            if (flag != STATUS_NOT_READY) {
                break;
            }
        }

        let flag = unpack_flag(predecessor_state);
        let value = unpack_value(predecessor_state);

        if (flag == STATUS_INCLUSIVE) {
            running_sum += value;
            break;
        } else {
            running_sum += value;
            lookback_id -= 1;
        }
    }

    exclusive_prefix = running_sum;

    let inclusive_value = exclusive_prefix + local_total;
    atomicStore(&state[wg_id], pack_state(STATUS_INCLUSIVE, inclusive_value));

    return exclusive_prefix;
}

// ============================================================================
// Main Kernel
// ============================================================================
@compute @workgroup_size(256)
fn decoupled_lookback_by_key_kernel(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tid = local_id.x;
    let block = get_workgroup_index(wg_id);

    if (block >= num_wg_total) {
        return;
    }

    // ========================================================================
    // Step 1: Read partition boundaries from DPI (thread 0)
    // ========================================================================
    if (tid == 0u) {
        let bp0 = dpi[block];
        let bp1 = dpi[block + 1u];

        let a0 = bp0 & INDEX_MASK;
        let a1 = bp1 & INDEX_MASK;

        let bit0 = select(0u, 1u, (bp0 & STAR_MASK) != 0u);
        let bit1 = select(0u, 1u, (bp1 & STAR_MASK) != 0u);

        let b0 = dpi[num_wg_total + 1u + block] + bit0;
        let b1 = dpi[num_wg_total + 1u + block + 1u] + bit1;

        let a_count2 = a1 - a0;
        let b_count2 = b1 - b0;
        let extended = (a1 < a_length) && (b1 < b_length);
        let b_start = a_count2 + 2u + select(0u, 1u, extended);

        wg_a0 = a0;
        wg_a1 = a1;
        wg_b0 = b0;
        wg_b1 = b1;
        wg_a_count = a_count2;
        wg_b_count = b_count2;
        wg_b_start = b_start;
        wg_extended = extended;
        wg_bit0 = bit0;
    }
    workgroupBarrier();

    let a0 = wg_a0;
    let b0 = wg_b0;
    let a_count2 = wg_a_count;
    let b_count2 = wg_b_count;
    let b_start = wg_b_start;
    let extended = wg_extended;
    let bit0 = wg_bit0;

    // ========================================================================
    // Step 2: Load data into shared memory with sentinels
    // ========================================================================
    let a_load_count = a_count2 + select(0u, 1u, extended);
    let b_load_count = b_count2 + select(0u, 1u, extended);

    device_load_2_to_shared(tid, a0, a_load_count, b0, b_load_count, b_start);
    workgroupBarrier();

    // ========================================================================
    // Step 3: Each thread finds its starting position using Local BalancedPath
    // ========================================================================
    let partition_size = a_count2 + b_count2;
    let diag_start = i32(VT * tid) - i32(bit0);
    let diag = min(diag_start, i32(partition_size));

    let bp = balanced_path_local(a_count2, b_start, b_count2, diag);

    let a0tid = bp.x;
    let star = bp.y;
    let b0tid_true = i32(VT * tid) + i32(star) - i32(a0tid) - i32(bit0);
    let b_adjust = u32(max(0, -b0tid_true));
    let b0tid = u32(max(0, b0tid_true));

    // ========================================================================
    // Step 4: Serial set operation by key
    // ========================================================================
    var result_keys_arr: array<u32, 12>;
    var result_values_arr: array<u32, 12>;

    let commit = serial_set_operation_by_key(
        a0tid,
        a_count2,
        b_start + b0tid,
        b_start + b_count2,
        b_start,         // b_start_base
        b0,              // b_global_offset
        star,
        b_adjust,
        extended,
        &result_keys_arr,
        &result_values_arr
    );

    // ========================================================================
    // Step 5: Workgroup-level exclusive scan to get local offsets and total
    // ========================================================================
    let local_count = countOneBits(commit);
    var wg_total: u32 = 0u;
    let local_offset = workgroup_exclusive_scan_with_total(tid, local_count, &wg_total);

    // ========================================================================
    // Step 6: Thread 0 performs Decoupled Lookback
    // ========================================================================
    if (tid == 0u) {
        wg_local_total = wg_total;
        wg_exclusive_prefix = decoupled_lookback(block, wg_total);

        // If this is the last workgroup, store the total count
        if (block == num_wg_total - 1u) {
            atomicStore(&total_count[0], wg_exclusive_prefix + wg_total);
        }
    }
    workgroupBarrier();

    // ========================================================================
    // Step 7: Scatter result keys and values to output - Unrolled (VT=12)
    // ========================================================================
    let global_offset = wg_exclusive_prefix + local_offset;

    var write_pos = global_offset;

    // Unrolled scatter loop - writes both key and value per committed element
    if ((commit & 0x01u) != 0u) { output_keys[write_pos] = result_keys_arr[0]; output_values[write_pos] = result_values_arr[0]; write_pos++; }
    if ((commit & 0x02u) != 0u) { output_keys[write_pos] = result_keys_arr[1]; output_values[write_pos] = result_values_arr[1]; write_pos++; }
    if ((commit & 0x04u) != 0u) { output_keys[write_pos] = result_keys_arr[2]; output_values[write_pos] = result_values_arr[2]; write_pos++; }
    if ((commit & 0x08u) != 0u) { output_keys[write_pos] = result_keys_arr[3]; output_values[write_pos] = result_values_arr[3]; write_pos++; }
    if ((commit & 0x10u) != 0u) { output_keys[write_pos] = result_keys_arr[4]; output_values[write_pos] = result_values_arr[4]; write_pos++; }
    if ((commit & 0x20u) != 0u) { output_keys[write_pos] = result_keys_arr[5]; output_values[write_pos] = result_values_arr[5]; write_pos++; }
    if ((commit & 0x40u) != 0u) { output_keys[write_pos] = result_keys_arr[6]; output_values[write_pos] = result_values_arr[6]; write_pos++; }
    if ((commit & 0x80u) != 0u) { output_keys[write_pos] = result_keys_arr[7]; output_values[write_pos] = result_values_arr[7]; write_pos++; }
    if ((commit & 0x100u) != 0u) { output_keys[write_pos] = result_keys_arr[8]; output_values[write_pos] = result_values_arr[8]; write_pos++; }
    if ((commit & 0x200u) != 0u) { output_keys[write_pos] = result_keys_arr[9]; output_values[write_pos] = result_values_arr[9]; write_pos++; }
    if ((commit & 0x400u) != 0u) { output_keys[write_pos] = result_keys_arr[10]; output_values[write_pos] = result_values_arr[10]; write_pos++; }
    if ((commit & 0x800u) != 0u) { output_keys[write_pos] = result_keys_arr[11]; output_values[write_pos] = result_values_arr[11]; write_pos++; }
}
