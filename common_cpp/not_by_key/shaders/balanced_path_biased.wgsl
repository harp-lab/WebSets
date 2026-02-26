// ============================================================================
// Balanced Path with Biased Binary Search (ModernGPU-style)
// Subgroup-optimized version: 256 threads/workgroup, each subgroup (warp)
// independently handles one diagonal using subgroup shuffle operations.
// Zero shared memory, zero workgroupBarrier() in the search loop.
//
// dpi layout:
//   dpi[0 .. num_wg]           : packed aIndex (MSB = star)
//   dpi[num_wg+1 .. 2*num_wg+1]: bIndex
//
// DISPATCH NOTE: The caller must dispatch
//   ceil(num_wg / (256 / subgroupSize)) workgroups
// instead of num_wg. For NVIDIA (subgroupSize=32): ceil(num_wg / 8).
// ============================================================================

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read_write> dpi: array<u32>;
@group(0) @binding(3) var<uniform> a_length: u32;
@group(0) @binding(4) var<uniform> b_length: u32;
@group(0) @binding(5) var<uniform> num_wg_uniform: u32;

const STAR_MASK: u32 = 0x80000000u;
const WORKGROUP_SIZE: u32 = 256u;

// ModernGPU constants for partition size
const NT: u32 = 256u;        // Threads per workgroup (in count kernel)
const VT: u32 = 12u;         // Values per thread (changed from 7 to match sentinel kernel)
const NV: u32 = NT * VT;     // Elements per workgroup = 3072

fn pack_a(a_idx: u32, star: bool) -> u32 {
    return select(a_idx, a_idx | STAR_MASK, star);
}

// ============================================================================
// Dynamic Levels Selection for Biased Binary Search
// ============================================================================
// The levels parameter controls how many biased steps to take before
// falling back to standard binary search.
//
// IMPORTANT: levels should be based on PARTITION SIZE
// The partition size represents the typical workload per workgroup:
//   partition_size = (a_length + b_length) / num_workgroups
//
// This determines the expected search range for finding duplicate runs.
//
// Level selection based on partition size:
//   partition_size >= 512  -> levels = 4 (shifts: 9,7,5,4,1)
//   partition_size >= 128  -> levels = 3 (shifts: 7,5,4,1)
//   partition_size >= 32   -> levels = 2 (shifts: 5,4,1)
//   partition_size >= 16   -> levels = 1 (shifts: 4,1)
//   partition_size < 16    -> levels = 0 (pure binary search)
//
// Larger partitions benefit from more aggressive biasing.
// ============================================================================
fn get_biased_search_levels(partition_size: u32) -> u32 {
    if (partition_size >= 512u) { return 4u; }
    if (partition_size >= 128u) { return 3u; }
    if (partition_size >= 32u)  { return 2u; }
    if (partition_size >= 16u)  { return 1u; }
    return 0u;
}

fn cmp_lte(x: u32, y: u32) -> bool {
    return x <= y;
}

// ============================================================================
// Biased Binary Search Implementation (ModernGPU-style)
// ============================================================================
//
// BiasedBinarySearch vs Standard Binary Search:
//
// Standard Binary Search:
//   - Always picks mid = (lo + hi) / 2
//   - Divides range in half each iteration
//   - O(log n) comparisons regardless of target position
//
// Biased Binary Search (ModernGPU formula):
//   - Formula: scale = (1 << shift) - 1, mid = (lo + scale * hi) >> shift
//   - Biases the search towards the END of the range
//   - shift=9: mid ~ hi - range/512  -> probe at ~99.8% of range (near end)
//   - shift=7: mid ~ hi - range/128  -> probe at ~99.2% of range
//   - shift=5: mid ~ hi - range/32   -> probe at ~97% of range
//   - shift=4: mid ~ hi - range/16   -> probe at ~94% of range
//   - shift=1: mid = (lo + hi) / 2   -> standard binary search
//
// Why bias towards END in BalancedPath?
//   When searching [0, aIndex) for the start of a duplicate run,
//   the run start is usually close to aIndex (short runs are common).
//   By probing near the end first, we find short runs faster.
//
// levels parameter:
//   Controls how many biased steps to take before standard binary search:
//   - levels=4: use shifts 9,7,5,4 then standard (most biased towards end)
//   - levels=3: use shifts 7,5,4 then standard
//   - levels=2: use shifts 5,4 then standard
//   - levels=1: use shift 4 then standard
//   - levels=0: pure standard binary search (no bias)
// ============================================================================

// Single biased binary search step for array A (lower_bound only)
// ModernGPU formula: scale = (1 << shift) - 1, mid = (lo + scale * hi) >> shift
// This biases the search towards the END of the range
fn binary_search_step_a(
    lo_ptr: ptr<function, u32>,
    hi_ptr: ptr<function, u32>,
    key: u32,
    shift: u32
) {
    let lo = *lo_ptr;
    let hi = *hi_ptr;
    if (lo >= hi) { return; }

    // ModernGPU biased midpoint (bias towards end)
    let scale = (1u << shift) - 1u;
    let mid = (lo + scale * hi) >> shift;

    // lower_bound: find first element >= key
    if (a[mid] < key) {
        *lo_ptr = mid + 1u;
    } else {
        *hi_ptr = mid;
    }
}

// Single biased binary search step for array B (lower_bound only)
fn binary_search_step_b(
    lo_ptr: ptr<function, u32>,
    hi_ptr: ptr<function, u32>,
    key: u32,
    shift: u32
) {
    let lo = *lo_ptr;
    let hi = *hi_ptr;
    if (lo >= hi) { return; }

    let scale = (1u << shift) - 1u;
    let mid = (lo + scale * hi) >> shift;

    // lower_bound: find first element >= key
    if (b[mid] < key) {
        *lo_ptr = mid + 1u;
    } else {
        *hi_ptr = mid;
    }
}

// Biased lower_bound for array A
// Searches [0, end_exclusive) for first element >= key
fn biased_lower_bound_a(end_exclusive: u32, key: u32, levels: u32) -> u32 {
    var lo: u32 = 0u;
    var hi: u32 = end_exclusive;

    // Biased probing steps (probing near the end first)
    if (levels >= 4u && lo < hi) { binary_search_step_a(&lo, &hi, key, 9u); }
    if (levels >= 3u && lo < hi) { binary_search_step_a(&lo, &hi, key, 7u); }
    if (levels >= 2u && lo < hi) { binary_search_step_a(&lo, &hi, key, 5u); }
    if (levels >= 1u && lo < hi) { binary_search_step_a(&lo, &hi, key, 4u); }

    // Standard binary search to finish
    while (lo < hi) {
        binary_search_step_a(&lo, &hi, key, 1u);
    }

    return lo;
}

// Biased lower_bound for array B
// Searches [0, end_exclusive) for first element >= key
fn biased_lower_bound_b(end_exclusive: u32, key: u32, levels: u32) -> u32 {
    var lo: u32 = 0u;
    var hi: u32 = end_exclusive;

    if (levels >= 4u && lo < hi) { binary_search_step_b(&lo, &hi, key, 9u); }
    if (levels >= 3u && lo < hi) { binary_search_step_b(&lo, &hi, key, 7u); }
    if (levels >= 2u && lo < hi) { binary_search_step_b(&lo, &hi, key, 5u); }
    if (levels >= 1u && lo < hi) { binary_search_step_b(&lo, &hi, key, 4u); }

    while (lo < hi) {
        binary_search_step_b(&lo, &hi, key, 1u);
    }

    return lo;
}

// Non-biased upper_bound for array B with custom range (used in balanced_adjust)
fn upper_bound_b(range_begin: u32, range_end: u32, key: u32) -> u32 {
    var lo: u32 = range_begin;
    var hi: u32 = range_end;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (b[mid] <= key) { lo = mid + 1u; } else { hi = mid; }
    }
    return lo;
}

// ============================================================================
// Balanced Path Adjustment (with Biased Binary Search)
// ============================================================================
// partition_size: the size of each partition (total_elements / num_workgroups)
//                 used to determine the biased search levels
fn balanced_adjust(a_count: u32, b_count: u32, diag: u32, p_in: u32, partition_size: u32) -> vec2<u32> {
    var p = p_in;
    if (p > diag) { p = diag; }

    var a_index = p;
    var b_index = diag - p;
    var star = false;

    if (b_index < b_count) {
        let x = b[b_index];

        // Use biased binary search to find start of duplicate run
        // Levels based on partition size (not current search range)
        let levels = get_biased_search_levels(partition_size);
        let a_start = biased_lower_bound_a(a_index, x, levels);
        let b_start = biased_lower_bound_b(b_index, x, levels);

        let a_run = a_index - a_start;
        let b_run = b_index - b_start;
        let x_count = a_run + b_run;

        // Split duplicate run evenly between partitions
        var b_advance = max(x_count >> 1u, x_count - a_run);

        // Probe upper bound to clamp b_advance
        var b_end_hint = min(b_count, b_start + b_advance + 1u);
        b_end_hint = max(b_end_hint, min(b_count, b_index + 1u));

        let b_run_end = upper_bound_b(b_index, b_end_hint, x);
        let actual_b_run = b_run_end - b_start;

        b_advance = min(b_advance, actual_b_run);
        let a_advance = x_count - b_advance;

        // Star marks odd split for correct handling at partition boundary
        let round_up = (a_advance == b_advance + 1u) && (b_advance < actual_b_run);

        a_index = a_start + a_advance;
        b_index = diag - a_index;
        star = round_up;
    }

    return vec2<u32>(pack_a(a_index, star), b_index);
}

// ============================================================================
// Helper: Find first set bit in subgroupBallot result
// Handles subgroup sizes up to 128 (vec4<u32> = 128 bits)
// ============================================================================
fn ballot_find_first(ballot: vec4<u32>, sg_sz: u32) -> u32 {
    if (ballot.x != 0u) {
        return countTrailingZeros(ballot.x);
    }
    if (sg_sz > 32u && ballot.y != 0u) {
        return 32u + countTrailingZeros(ballot.y);
    }
    if (sg_sz > 64u && ballot.z != 0u) {
        return 64u + countTrailingZeros(ballot.z);
    }
    if (sg_sz > 96u && ballot.w != 0u) {
        return 96u + countTrailingZeros(ballot.w);
    }
    return sg_sz; // not found
}

// ============================================================================
// Subgroup-optimized compute_diagonals
//
// Each subgroup (warp) independently processes one diagonal.
// 256 threads / subgroup_size subgroups per workgroup.
// All communication via subgroup shuffle — zero shared memory.
// ============================================================================
@compute @workgroup_size(WORKGROUP_SIZE)
fn compute_diagonals(@builtin(global_invocation_id) global_id: vec3<u32>,
                     @builtin(num_workgroups) n_wgs: vec3<u32>,
                     @builtin(local_invocation_id) local_id: vec3<u32>,
                     @builtin(workgroup_id) wg: vec3<u32>,
                     @builtin(subgroup_invocation_id) sg_lane: u32,
                     @builtin(subgroup_size) sg_size: u32) {

    let local_flat = local_id.x;
    let group_flat = wg.y * n_wgs.x + wg.x;
    let num_wg = num_wg_uniform;

    // Each subgroup independently handles one diagonal
    let subgroup_id = local_flat / sg_size;
    let subgroups_per_wg = WORKGROUP_SIZE / sg_size;
    let k = group_flat * subgroups_per_wg + subgroup_id;

    // Write boundary diagonals (once, by lane 0 of the subgroup with k==0)
    if (sg_lane == 0u && k == 0u) {
        dpi[0u] = pack_a(0u, false);
        dpi[num_wg + 1u] = 0u;
        dpi[num_wg] = pack_a(a_length, false);
        dpi[num_wg + num_wg + 1u] = b_length;
    }

    // Skip boundary and out-of-range diagonals
    if (k == 0u || k >= num_wg) { return; }

    // ModernGPU diagonal calculation: gid = NV * block
    let diag_u: u32 = NV * k;

    let combined_index: i32 = i32(diag_u);
    let n_a: i32 = i32(a_length);
    let n_b: i32 = i32(b_length);

    // Initialize search window in registers (uniform across all lanes in subgroup)
    var xt: i32 = min(combined_index, n_a);
    var yt: i32 = max(0, combined_index - n_a);
    var xb: i32 = max(0, combined_index - n_b);
    var yb: i32 = min(combined_index, n_b);

    // Each lane samples a different point on the diagonal
    let half_sg = i32(sg_size >> 1u);
    let thread_offset = i32(sg_lane) - half_sg;

    var fx: u32 = 0u;
    var fy: u32 = 0u;

    // Parallel MergePath search using subgroup operations (no shared memory)
    loop {
        let current_x = xt - ((xt - xb) >> 1) - thread_offset;
        let current_y = yt + ((yb - yt) >> 1) + thread_offset;

        var r: u32 = 0u;
        if (current_x > n_a || current_y < 0) {
            r = 0u;
        } else if (current_y >= n_b || current_x < 1) {
            r = 1u;
        } else {
            r = select(0u, 1u, cmp_lte(a[u32(current_x - 1)], b[u32(current_y)]));
        }

        // Boundary detection: compare with previous lane via subgroupShuffleUp
        let prev_r = subgroupShuffleUp(r, 1u);
        let is_boundary = (sg_lane > 0u) && (r != prev_r);

        // Find first 0->1 boundary using ballot
        let ballot = subgroupBallot(is_boundary);
        let first_bit = ballot_find_first(ballot, sg_size);

        if (first_bit < sg_size) {
            // Broadcast found position from boundary lane to all lanes
            fx = subgroupShuffle(u32(current_x), first_bit);
            fy = subgroupShuffle(u32(current_y), first_bit);
            break;
        }

        // Shrink search window: read from midpoint lane and last lane
        let val_last = subgroupShuffle(r, sg_size - 1u);
        let cx_mid = i32(subgroupShuffle(u32(current_x), u32(half_sg)));
        let cy_mid = i32(subgroupShuffle(u32(current_y), u32(half_sg)));

        if (val_last != 0u) {
            xb = cx_mid;
            yb = cy_mid;
        } else {
            xt = cx_mid;
            yt = cy_mid;
        }
    }

    // Only lane 0 of each subgroup writes the result
    if (sg_lane == 0u) {
        // ModernGPU uses fixed partition size NV = 3072
        let out = balanced_adjust(a_length, b_length, diag_u, fx, NV);
        dpi[k] = out.x;
        dpi[k + num_wg + 1u] = out.y;
    }
}
