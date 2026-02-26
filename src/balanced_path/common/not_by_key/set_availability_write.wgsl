// ============================================================================
// Unified Set Availability - Write Kernel (Sentinel + VT=12)
//
// ModernGPU-style DeviceComputeSetAvailability implementation
// Two-pass approach: Write phase - re-executes SerialSetOp and scatters results
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
// 2. Load A and B data into shared memory with sentinels
// 3. Each thread runs Local BalancedPath to find its starting position
// 4. Each thread runs SerialSetOperation to get results
// 5. Workgroup exclusive scan to compute local offsets
// 6. Scatter results to output using global offset + local offset
//
// NOTE: This kernel MUST use identical commit logic to the count kernel!
// ============================================================================

// ============================================================================
// Operation Mode (override via string replacement in TypeScript)
//   0 = intersection, 1 = difference, 2 = union, 3 = sym_difference
// ============================================================================
const OP_MODE: u32 = 0u;

// ============================================================================
// Bindings (8 total)
// ============================================================================
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read> dpi: array<u32>;
@group(0) @binding(3) var<storage, read> offsets: array<u32>;  // Prefix sum of counts
@group(0) @binding(4) var<storage, read_write> output: array<u32>;
@group(0) @binding(5) var<uniform> a_length: u32;
@group(0) @binding(6) var<uniform> b_length: u32;
@group(0) @binding(7) var<uniform> num_wg_total: u32;

// ============================================================================
// Constants (ModernGPU terminology)
// ============================================================================
const NT: u32 = 256u;           // Threads per workgroup
const VT: u32 = 12u;            // Values per thread
const NV: u32 = NT * VT;        // Total elements per workgroup = 3072

const STAR_MASK: u32 = 0x80000000u;
const INDEX_MASK: u32 = 0x7FFFFFFFu;
const MAX_DISPATCH_X: u32 = 65535u;

// Sentinel values
const POS_INF: u32 = 0xFFFFFFFFu;
const NEG_INF: u32 = 0u;

// ============================================================================
// Shared Memory Layout with Sentinels
// ============================================================================
// keys_shared: A keys + B keys with sentinel slots
//   keys_shared[0]                               = NEG_INF (A leading sentinel)
//   keys_shared[1 .. a_count+1)                  = A data (offset +1)
//   keys_shared[a_count+1]                       = POS_INF (A trailing sentinel)
//   keys_shared[b_start]                         = NEG_INF (B leading sentinel)
//   keys_shared[b_start+1 .. b_start+b_count+1)  = B data (offset +1)
//   keys_shared[b_start+b_count+1]               = POS_INF (B trailing sentinel)
//
// Total size: NV + VT + 2 + 4 = 3090
var<workgroup> keys_shared: array<u32, 3090>;

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

// For workgroup exclusive scan
var<workgroup> shared_scan: array<u32, NT>;

// ============================================================================
// Helper: Convert 2D workgroup_id to 1D index
// ============================================================================
fn get_workgroup_index(wg_id: vec3<u32>) -> u32 {
    return wg_id.x + wg_id.y * MAX_DISPATCH_X;
}

// ============================================================================
// DeviceLoad2ToShared: Cooperative loading with sentinels
// Data is offset by +1 to make room for leading sentinel
// ============================================================================
fn device_load_2_to_shared(
    tid: u32,
    a_global_offset: u32,
    a_load_count: u32,
    b_global_offset: u32,
    b_load_count: u32,
    b_shared_start: u32
) {
    // Write sentinels (only specific threads to avoid redundant writes)
    if (tid == 0u) {
        keys_shared[0] = NEG_INF;                        // A leading sentinel
        keys_shared[a_load_count + 1u] = POS_INF;        // A trailing sentinel
        keys_shared[b_shared_start] = NEG_INF;           // B leading sentinel
        keys_shared[b_shared_start + b_load_count + 1u] = POS_INF;  // B trailing sentinel
    }

    // Load A elements: unrolled 12 iterations, with +1 offset for sentinel
    if (tid < a_load_count) {
        keys_shared[tid + 1u] = a[a_global_offset + tid];
    }
    if (tid + 256u < a_load_count) {
        keys_shared[tid + 257u] = a[a_global_offset + tid + 256u];
    }
    if (tid + 512u < a_load_count) {
        keys_shared[tid + 513u] = a[a_global_offset + tid + 512u];
    }
    if (tid + 768u < a_load_count) {
        keys_shared[tid + 769u] = a[a_global_offset + tid + 768u];
    }
    if (tid + 1024u < a_load_count) {
        keys_shared[tid + 1025u] = a[a_global_offset + tid + 1024u];
    }
    if (tid + 1280u < a_load_count) {
        keys_shared[tid + 1281u] = a[a_global_offset + tid + 1280u];
    }
    if (tid + 1536u < a_load_count) {
        keys_shared[tid + 1537u] = a[a_global_offset + tid + 1536u];
    }
    if (tid + 1792u < a_load_count) {
        keys_shared[tid + 1793u] = a[a_global_offset + tid + 1792u];
    }
    if (tid + 2048u < a_load_count) {
        keys_shared[tid + 2049u] = a[a_global_offset + tid + 2048u];
    }
    if (tid + 2304u < a_load_count) {
        keys_shared[tid + 2305u] = a[a_global_offset + tid + 2304u];
    }
    if (tid + 2560u < a_load_count) {
        keys_shared[tid + 2561u] = a[a_global_offset + tid + 2560u];
    }
    if (tid + 2816u < a_load_count) {
        keys_shared[tid + 2817u] = a[a_global_offset + tid + 2816u];
    }

    // Load B elements: unrolled 12 iterations, with +1 offset for sentinel
    if (tid < b_load_count) {
        keys_shared[b_shared_start + tid + 1u] = b[b_global_offset + tid];
    }
    if (tid + 256u < b_load_count) {
        keys_shared[b_shared_start + tid + 257u] = b[b_global_offset + tid + 256u];
    }
    if (tid + 512u < b_load_count) {
        keys_shared[b_shared_start + tid + 513u] = b[b_global_offset + tid + 512u];
    }
    if (tid + 768u < b_load_count) {
        keys_shared[b_shared_start + tid + 769u] = b[b_global_offset + tid + 768u];
    }
    if (tid + 1024u < b_load_count) {
        keys_shared[b_shared_start + tid + 1025u] = b[b_global_offset + tid + 1024u];
    }
    if (tid + 1280u < b_load_count) {
        keys_shared[b_shared_start + tid + 1281u] = b[b_global_offset + tid + 1280u];
    }
    if (tid + 1536u < b_load_count) {
        keys_shared[b_shared_start + tid + 1537u] = b[b_global_offset + tid + 1536u];
    }
    if (tid + 1792u < b_load_count) {
        keys_shared[b_shared_start + tid + 1793u] = b[b_global_offset + tid + 1792u];
    }
    if (tid + 2048u < b_load_count) {
        keys_shared[b_shared_start + tid + 2049u] = b[b_global_offset + tid + 2048u];
    }
    if (tid + 2304u < b_load_count) {
        keys_shared[b_shared_start + tid + 2305u] = b[b_global_offset + tid + 2304u];
    }
    if (tid + 2560u < b_load_count) {
        keys_shared[b_shared_start + tid + 2561u] = b[b_global_offset + tid + 2560u];
    }
    if (tid + 2816u < b_load_count) {
        keys_shared[b_shared_start + tid + 2817u] = b[b_global_offset + tid + 2816u];
    }
}

// ============================================================================
// Local Balanced Path with Biased Binary Search (ModernGPU Style)
//
// With sentinel optimization: indices are +1 offset (data starts at index 1)
// ============================================================================

// Get biased search levels based on partition size (matches global version)
fn get_local_biased_levels(partition_size: u32) -> u32 {
    if (partition_size >= 512u) { return 4u; }
    if (partition_size >= 128u) { return 3u; }
    if (partition_size >= 32u)  { return 2u; }
    if (partition_size >= 16u)  { return 1u; }
    return 0u;
}

// Biased lower_bound for A in shared memory (with sentinel offset)
// Searches logical [0, end_exclusive) which maps to physical [1, end_exclusive+1)
fn lower_bound_local_a(end_exclusive: u32, key: u32, levels: u32) -> u32 {
    var lo: u32 = 0u;
    var hi: u32 = end_exclusive;

    // Biased probing steps (probing near the end first)
    // Physical access: keys_shared[lo/hi/mid + 1]
    if (levels >= 4u && lo < hi) {
        let scale = (1u << 9u) - 1u;
        let mid = (lo + scale * hi) >> 9u;
        if (keys_shared[mid + 1u] < key) { lo = mid + 1u; } else { hi = mid; }
    }
    if (levels >= 3u && lo < hi) {
        let scale = (1u << 7u) - 1u;
        let mid = (lo + scale * hi) >> 7u;
        if (keys_shared[mid + 1u] < key) { lo = mid + 1u; } else { hi = mid; }
    }
    if (levels >= 2u && lo < hi) {
        let scale = (1u << 5u) - 1u;
        let mid = (lo + scale * hi) >> 5u;
        if (keys_shared[mid + 1u] < key) { lo = mid + 1u; } else { hi = mid; }
    }
    if (levels >= 1u && lo < hi) {
        let scale = (1u << 4u) - 1u;
        let mid = (lo + scale * hi) >> 4u;
        if (keys_shared[mid + 1u] < key) { lo = mid + 1u; } else { hi = mid; }
    }

    // Standard binary search to finish
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (keys_shared[mid + 1u] < key) { lo = mid + 1u; } else { hi = mid; }
    }

    return lo;
}

// Biased lower_bound for B in shared memory (with sentinel offset)
// Searches logical [0, end_exclusive) which maps to physical [b_start+1, b_start+end_exclusive+1)
fn lower_bound_local_b(b_start: u32, end_exclusive: u32, key: u32, levels: u32) -> u32 {
    var lo: u32 = 0u;
    var hi: u32 = end_exclusive;

    // Physical access: keys_shared[b_start + lo/hi/mid + 1]
    if (levels >= 4u && lo < hi) {
        let scale = (1u << 9u) - 1u;
        let mid = (lo + scale * hi) >> 9u;
        if (keys_shared[b_start + mid + 1u] < key) { lo = mid + 1u; } else { hi = mid; }
    }
    if (levels >= 3u && lo < hi) {
        let scale = (1u << 7u) - 1u;
        let mid = (lo + scale * hi) >> 7u;
        if (keys_shared[b_start + mid + 1u] < key) { lo = mid + 1u; } else { hi = mid; }
    }
    if (levels >= 2u && lo < hi) {
        let scale = (1u << 5u) - 1u;
        let mid = (lo + scale * hi) >> 5u;
        if (keys_shared[b_start + mid + 1u] < key) { lo = mid + 1u; } else { hi = mid; }
    }
    if (levels >= 1u && lo < hi) {
        let scale = (1u << 4u) - 1u;
        let mid = (lo + scale * hi) >> 4u;
        if (keys_shared[b_start + mid + 1u] < key) { lo = mid + 1u; } else { hi = mid; }
    }

    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (keys_shared[b_start + mid + 1u] < key) { lo = mid + 1u; } else { hi = mid; }
    }

    return lo;
}

// Upper_bound for B in shared memory (with sentinel offset)
// Searches logical [range_begin, range_end) which maps to physical [b_start+range_begin+1, b_start+range_end+1)
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
// Returns the A index (p) where the merge path crosses the diagonal
fn merge_path_local(a_count: u32, b_start: u32, b_count: u32, diag: u32) -> u32 {
    // With sentinels, we can simplify boundary initialization
    // The sentinels guarantee valid accesses even at boundaries
    var lo: u32 = select(0u, diag - b_count, diag > b_count);
    var hi: u32 = min(diag, a_count);

    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        // Physical access: A at [mid + 1], B at [b_start + (diag - 1 - mid) + 1]
        let a_key = keys_shared[mid + 1u];
        let b_key = keys_shared[b_start + diag - mid];  // diag - 1 - mid + 1 = diag - mid
        if (a_key <= b_key) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }

    return lo;
}

// Complete Balanced Path with adjustment (matches global balanced_adjust)
// Returns (a_index, star) - ModernGPU style
fn balanced_path_local_biased(
    a_count: u32,
    b_start: u32,
    b_count: u32,
    diag: i32
) -> vec2<u32> {
    let diag_u = u32(max(0, diag));

    // Handle boundary cases
    if (diag_u == 0u) {
        return vec2<u32>(0u, 0u);  // (a_index=0, star=0)
    }
    if (diag_u >= a_count + b_count) {
        return vec2<u32>(a_count, 0u);  // (a_index=a_count, star=0)
    }

    // Step 1: Basic MergePath search
    let p = merge_path_local(a_count, b_start, b_count, diag_u);

    var a_index = p;
    var b_index = diag_u - p;
    var star: u32 = 0u;

    // Step 2: Balanced adjustment for duplicates
    if (b_index < b_count) {
        // Physical access: keys_shared[b_start + b_index + 1]
        let x = keys_shared[b_start + b_index + 1u];
        let levels = get_local_biased_levels(VT);
        let a_start = lower_bound_local_a(a_index, x, levels);
        let b_start_run = lower_bound_local_b(b_start, b_index, x, levels);
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

        // Calculate star bit: marks odd split for correct boundary handling
        let round_up = (a_advance == b_advance + 1u) && (b_advance < actual_b_run);
        star = select(0u, 1u, round_up);

        a_index = a_start + a_advance;
    }

    return vec2<u32>(a_index, star);
}

// ============================================================================
// Serial Set Operation (ModernGPU Style) - with sentinel offset
//
// Unified serial function supporting all 4 set operations via OP_MODE.
// Branchless select() for pointer advancement + pre-fetch pattern.
// Physical indices are +1 from logical indices.
//
// OP_MODE commit conditions:
//   0 (intersection): pA == pB  (both false → A == B → emit)
//   1 (difference):   pA        (A < B → emit A, not in B)
//   2 (union):        true      (always emit)
//   3 (sym_diff):     pA != pB  (exactly one true -> emit)
//
// OP_MODE result values:
//   0,1 (intersection, difference): always a_key
//   2,3 (union, sym_difference):    select(a_key, b_key, pB)
// ============================================================================
fn serial_set_operation(
    a_begin: u32,
    a_end: u32,
    b_begin: u32,
    b_end: u32,
    star: u32,
    b_adjust: u32,
    extended: bool,
    results: ptr<function, array<u32, 12>>,
    indices: ptr<function, array<u32, 12>>
) -> u32 {
    var commit: u32 = 0u;
    var a_idx = a_begin;
    var b_idx = b_begin;

    // ModernGPU: end = aBegin + bBegin + VT - star
    var end_diag = i32(a_begin) + i32(b_begin) + i32(VT) - i32(star) - i32(b_adjust);
    if (!extended) {
        end_diag = min(end_diag, i32(a_end) + i32(b_end));
    }

    let min_iterations = VT / 2u;

    // Pre-fetch first keys (physical indices have +1 offset)
    // a_idx is logical, physical is a_idx + 1
    // b_idx is already physical (includes b_start offset), so we add 1 for sentinel
    var a_key = keys_shared[a_idx + 1u];
    var b_key = keys_shared[b_idx + 1u];

    for (var i: u32 = 0u; i < VT; i++) {
        // ModernGPU termination condition
        var test: bool;
        if (extended) {
            test = (i < min_iterations) || (i32(a_idx) + i32(b_idx) < end_diag);
        } else if (OP_MODE == 0u) {
            // intersection: can also use explicit bounds (redundant with sentinels)
            test = (i32(a_idx) + i32(b_idx) < end_diag) && (a_idx < a_end) && (b_idx < b_end);
        } else {
            // difference/union/sym_diff: sentinels handle exhaustion,
            // must NOT early-exit when one array is exhausted
            test = (i32(a_idx) + i32(b_idx) < end_diag);
        }

        if (!test) {
            break;
        }

        let pA = a_key < b_key;
        let pB = b_key < a_key;

        // ============ OP_MODE: result value ============
        if (OP_MODE <= 1u) {
            // intersection, difference: always output A
            (*results)[i] = a_key;
            (*indices)[i] = a_idx;
        } else {
            // union, sym_difference: output B when B < A
            (*results)[i] = select(a_key, b_key, pB);
            (*indices)[i] = select(a_idx, b_idx, pB);
        }

        // ============ OP_MODE: commit condition ============
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
    }

    return commit;
}

// ============================================================================
// popcount for commit bitmask
// ============================================================================
fn countOneBits(n_in: u32) -> u32 {
    var n = n_in;
    n = n - ((n >> 1u) & 0x55555555u);
    n = (n & 0x33333333u) + ((n >> 2u) & 0x33333333u);
    return (((n + (n >> 4u)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24u;
}

// ============================================================================
// Workgroup exclusive scan (Hillis-Steele style)
// Returns the exclusive prefix sum for this thread
// ============================================================================
fn workgroup_exclusive_scan(tid: u32, value: u32) -> u32 {
    shared_scan[tid] = value;
    workgroupBarrier();

    // Up-sweep phase
    for (var offset: u32 = 1u; offset < NT; offset *= 2u) {
        var temp: u32 = 0u;
        if (tid >= offset) {
            temp = shared_scan[tid - offset];
        }
        workgroupBarrier();
        shared_scan[tid] += temp;
        workgroupBarrier();
    }

    // Convert to exclusive scan
    workgroupBarrier();

    if (tid == 0u) {
        return 0u;
    } else {
        return shared_scan[tid - 1u];
    }
}

// ============================================================================
// Main Kernel: Write Set Operation Results
// ============================================================================
@compute @workgroup_size(256)
fn write_availability(
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
        // b_start needs +2 for sentinel space: +1 for A trailing sentinel, +1 for B leading sentinel
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
    let total_count = a_count2 + b_count2;

    // Calculate diagonal range for this thread (with bit0 adjustment)
    let diag_start = i32(VT * tid) - i32(bit0);
    let diag = min(diag_start, i32(total_count));

    // BalancedPath returns (a_index, star) - ModernGPU style
    // These are logical indices (0-based into the data, not counting sentinels)
    let bp = balanced_path_local_biased(a_count2, b_start, b_count2, diag);

    // Thread starting positions (ModernGPU formula)
    let a0tid = bp.x;
    let star = bp.y;
    let b0tid_true = i32(VT * tid) + i32(star) - i32(a0tid) - i32(bit0);
    let b_adjust = u32(max(0, -b0tid_true));
    let b0tid = u32(max(0, b0tid_true));

    // ========================================================================
    // Step 4: Serial set operation (re-execute to get results)
    // ========================================================================
    var results: array<u32, 12>;
    var indices: array<u32, 12>;

    let commit = serial_set_operation(
        a0tid,
        a_count2,
        b_start + b0tid,
        b_start + b_count2,
        star,
        b_adjust,
        extended,
        &results,
        &indices
    );

    // ========================================================================
    // Step 5: Workgroup exclusive scan to compute local offsets
    // ========================================================================
    let local_count = countOneBits(commit);
    let local_offset = workgroup_exclusive_scan(tid, local_count);

    // ========================================================================
    // Step 6: Scatter results to output
    // ========================================================================
    let global_offset = offsets[block] + local_offset;

    var write_pos = global_offset;
    for (var i: u32 = 0u; i < VT; i++) {
        if ((commit & (1u << i)) != 0u) {
            output[write_pos] = results[i];
            write_pos++;
        }
    }
}
