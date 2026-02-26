const BLOCK_SIZE : u32 = 512u;

struct Data {
    vals : array<u32>,
};

struct BlockSums {
    block_sums : array<u32>,
};

@group(0) @binding(0) var<storage, read_write> data : Data;
@group(0) @binding(1) var<storage, read_write> sums : BlockSums;

var<workgroup> chunk: array<u32, BLOCK_SIZE>;

@compute @workgroup_size(BLOCK_SIZE / 2u)
fn main(
    @builtin(global_invocation_id)  global_id  : vec3<u32>,
    @builtin(local_invocation_id)   local_id   : vec3<u32>,
    @builtin(workgroup_id)          wg_id      : vec3<u32>
) {

    chunk[2u * local_id.x] = data.vals[2u * global_id.x];
    chunk[2u * local_id.x + 1u] = data.vals[2u * global_id.x + 1u];

    var offs : u32 = 1u;
    // Reduce step up tree
    for(var d : u32 = BLOCK_SIZE >> 1u; d > 0u; d = d >> 1u) {
        workgroupBarrier();
        if (local_id.x < d) {
            let a : u32 = offs * (2u * local_id.x + 1u) - 1u;
            let b : u32 = offs * (2u * local_id.x + 2u) - 1u;
            chunk[b] = chunk[b] + chunk[a];
        }
        offs = offs << 1u;
    }

    if (local_id.x == 0u) {
        sums.block_sums[wg_id.x] = chunk[BLOCK_SIZE - 1u];
        chunk[BLOCK_SIZE - 1u] = 0u;
    }

    // Sweep down the tree to finish the scan
    for (var d : u32 = 1u; d < BLOCK_SIZE; d = d << 1u) {
        offs = offs >> 1u;
        workgroupBarrier();
        if (local_id.x < d) {
            let a : u32 = offs * (2u * local_id.x + 1u) - 1u;
            let b : u32 = offs * (2u * local_id.x + 2u) - 1u;
            let tmp : u32 = chunk[a];
            chunk[a] = chunk[b];
            chunk[b] = chunk[b] + tmp;
        }
    }

    workgroupBarrier();
    data.vals[2u * global_id.x] = chunk[2u * local_id.x];
    data.vals[2u * global_id.x + 1u] = chunk[2u * local_id.x + 1u];
}
