# WebSets

High-performance GPU-accelerated set operations on large datasets using WebGPU. Supports intersection, union, difference, and symmetric difference on sorted multisets with up to 128 million elements.

## Features

- **4 set operations**: intersection (A &cap; B), difference (A \ B), union (A &cup; B), symmetric difference ((A \ B) &cup; (B \ A))
- **Not-by-key and by-key variants**: not-by-key operates on plain integer sets; by-key compares by keys but preserves key-value pairs in output
- **Two GPU pipeline architectures**:
  - **4-Phase**: DPI &rarr; Count &rarr; Prefix Sum &rarr; Write (uses separate scan pass)
  - **2-Phase**: DPI &rarr; Decoupled Lookback (single-pass count + scan + write)
- **Two implementations**:
  - **WebGPU (Browser)**: TypeScript + WGSL, runs in Chrome with WebGPU enabled
  - **C++ Native**: wgpu-native v24, runs as a standalone executable via Vulkan/D3D12/Metal backend

## Repository Structure

```
WebSets/
├── src/                                    # WebGPU (Browser) implementation
│   ├── app.ts                              # Main entry point
│   ├── utils.ts                            # Data loading and GPU buffer utilities
│   ├── TimestampQueryManager.ts            # GPU timestamp query profiling
│   ├── wgsl.d.ts                           # TypeScript declarations for .wgsl imports
│   └── balanced_path/common/
│       ├── balanced_path_biased.wgsl       # DPI kernel (subgroup-optimized)
│       ├── set_availability_count.wgsl     # 4-Phase count kernel
│       ├── not_by_key/                     # Not-by-key shaders and test harness
│       │   ├── set_availability_decoupled_lookback.wgsl
│       │   ├── set_availability_write.wgsl
│       │   └── test_unified_4phase_vs_2phase.ts
│       ├── by_key/                         # By-key shaders and test harness
│       │   ├── set_availability_decoupled_lookback_by_key.wgsl
│       │   ├── set_availability_write_by_key.wgsl
│       │   └── test_unified_4phase_vs_2phase_by_key.ts
│       ├── prefix_sum/                     # Multi-level exclusive scan
│       └── radix_sort/                     # GPU radix sort for unsorted inputs
│
├── common_cpp/                             # C++ Native (wgpu-native) implementation
│   ├── not_by_key/                         # Not-by-key: unified 2-Phase pipeline
│   │   ├── CMakeLists.txt
│   │   ├── main.cpp
│   │   ├── shaders/
│   │   └── webgpu/                         # WebGPU CMake distribution
│   └── by_key/                             # By-key: unified 2-Phase pipeline
│       ├── CMakeLists.txt
│       ├── main.cpp
│       ├── shaders/
│       └── webgpu/
│
├── index.html                              # HTML entry point
├── package.json                            # npm dependencies
├── tsconfig.json                           # TypeScript configuration
└── webpack.config.js                       # Webpack build configuration
```

## Algorithm Overview

The core algorithm is based on **Balanced Path** (ModernGPU-style merge path partitioning) combined with either a traditional multi-pass pipeline or a single-pass **Decoupled Lookback** scan.

### 2-Phase Pipeline (Recommended)

1. **DPI (Diagonal Path Intersection)**: Subgroup-optimized kernel that computes merge path partition boundaries across both input arrays. Each subgroup (warp) independently handles one diagonal using shuffle operations with zero shared memory.

2. **Decoupled Lookback**: Single-pass kernel that combines counting, prefix sum, and output writing. Each workgroup loads its partition into shared memory with sentinel values, performs a local serial set operation (VT=12 elements per thread), computes a workgroup-level scan, and uses the decoupled lookback protocol to obtain global output offsets without a separate scan pass.

All 4 set operations share the same pipeline code, differing only by a compile-time `OP_MODE` constant (0-3) that controls which elements are emitted.

## Getting Started

### Prerequisites

- **Browser version**: Chrome 113+ or Edge 113+ with WebGPU enabled
- **C++ version**: CMake 3.0+, C++17 compiler, internet connection (for wgpu-native download via FetchContent)

### Test Data

Both implementations expect sorted binary data files in the `public/data/` directory:

- Naming: `A_<size>e<valueRange>.bin` (e.g., `A_1e2.bin` = 1M elements, value range 10^2)
- **Download**: [Google Drive](https://drive.google.com/file/d/1bMNS455RsbOzQ86oQCstvxxRmhqBpKPh/view?usp=sharing) — extract to `public/data/`

### WebGPU (Browser)

```bash
# Install dependencies
npm install

# Start development server
npm run serve

# Open http://localhost:8080 and check the browser console for results
```

Edit `src/app.ts` to select which test to run by uncommenting the desired function call:

```typescript
// Not-by-key tests
await runUnifiedSingleOpTest(device, 0);  // intersection
await runUnifiedSingleOpTest(device, 1);  // difference
await runUnifiedSingleOpTest(device, 2);  // union
await runUnifiedSingleOpTest(device, 3);  // symmetric difference

// By-key tests
await runUnifiedByKeySingleOpTest(device, 0);  // intersection by key
await runUnifiedByKeySingleOpTest(device, 1);  // difference by key
await runUnifiedByKeySingleOpTest(device, 2);  // union by key
await runUnifiedByKeySingleOpTest(device, 3);  // symmetric difference by key
```

### C++ Native (wgpu-native)

```bash
# Build not-by-key
cd common_cpp/not_by_key
mkdir build && cd build
cmake ..
cmake --build . --config Release

# Run (from the build directory)
./Release/App      # Linux/macOS
.\Release\App.exe  # Windows

# Build by-key (same steps)
cd common_cpp/by_key
mkdir build && cd build
cmake .. && cmake --build . --config Release
./Release/App
```

The C++ executable will:
1. Initialize WebGPU via wgpu-native (auto-selects Vulkan/D3D12/Metal backend)
2. Validate all 4 operations against CPU reference on `1e2` dataset
3. Benchmark all datasets with GPU timestamp queries (10 warmup + 100 measured iterations)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
