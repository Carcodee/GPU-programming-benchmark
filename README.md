# GPU Programming Benchmark

A focused GPU performance study exploring how different OpenCL kernel designs and different local work group sizes affect throughput, occupancy, and scaling behavior on modern GPUs.  
This benchmark aims to reveal how shared memory pressure, thread organization, and hardware scheduling influence real GPU performance.

---

## Overview

This project compares two main scenarios:

1. **High shared-memory usage** (reduced occupancy)  
2. **Zero shared-memory usage** (maximum occupancy)

By sweeping the global size for each local size configuration, the benchmark exposes where the GPU saturates, where performance plateaus, and how internal SM/compute-unit limits influence execution time.

The repository contains both the Python benchmarking script and generated plots.

---

## Tested Kernels

### 1. reduce_occupancy (shared memory heavy)

This kernel allocates a large local memory buffer, forcing the GPU to reserve substantial shared memory per block and reducing simultaneous block residency.

```c
__kernel void reduce_occupancy(__global float *a)
{
     __local float shared[1024 * 11];
     barrier(CLK_LOCAL_MEM_FENCE);

     int lid = get_local_id(0);
     for(int i = 0; i < 1024 * 11; i++){
        shared[lid] *= lid * 2.0f;
     }

     float temp = 1.0f;
     for (int i = 0; i < 1000000; ++i) {
        temp *= get_local_id(0);
     }

     int gid = get_global_id(0);
     a[gid] = temp;
}
```
This kernel stresses local memory usage, forcing each work group to reserve roughly 45 KB of shared memory.
On GPUs with limited shared memory per SM, this significantly reduces occupancy and exposes how shared memory pressure affects performance.

2. increase_occupancy (ALU-only baseline)
This kernel performs the same arithmetic loop but avoids using local memory completely, allowing maximum parallelism.

```
__kernel void increase_occupancy(__global float *a)
{
     float temp = 1.0f;
     for (int i = 0; i < 1000000; ++i) {
        temp *= get_local_id(0);
     }

     int gid = get_global_id(0);
     a[gid] = temp;
}
```

Because no shared memory is allocated, the GPU scheduler can keep far more work groups resident simultaneously.
This kernel serves as the baseline for maximum occupancy.

##Benchmark Methodology
For each local_size, the Python script:

- Starts with a global size equal to the largest local size.
- Increases global size proportionally to the local size.
- Allocates OpenCL buffers.
- Enqueues the kernel through PyOpenCL with profiling enabled.
- Records GPU execution time.
- Repeats for every local size and stores all results.

##Example snippet:

```python
Copy code
local_sizes = np.array([1, 2, 4, 8, 16, 32, 64])
global_size_arr = np.zeros((len(local_sizes), task_size), dtype=int)
elapsed_times = np.zeros((len(local_sizes), task_size), dtype=float)
Each local size produces a full timing curve, and all curves are plotted together to compare occupancy behavior.
```

##What You Learn From This Benchmark
- How shared memory allocation reduces SM occupancy.

- Why small local sizes limit SIMD utilization and often hurt throughput.

- How large local sizes can saturate compute units but may exceed per-SM limits.

- When the GPU reaches full occupancy and why performance stops scaling.

- How global size interacts with work group size to match or mismatch the GPU hardware scheduling model.

##Requirements
- OpenCL 1.2 or newer
- Python 3.x
- PyOpenCL
- NumPy
- Matplotlib
