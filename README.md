GPU Programming Benchmark

This project is a focused performance study that evaluates how different OpenCL kernel designs and different local work group sizes influence overall GPU throughput. The goal is to understand the internal behavior of modern GPUs at a low level. This includes how cores, streaming multiprocessors, execution units, thread blocks and local memory usage interact to determine occupancy and performance.

The figures shown in this repository visualize the execution time of two kernels, each exploring a different resource usage scenario. By sweeping the global work size while holding the local size fixed, the benchmark reveals how the GPU scales, how occupancy changes, and how shared memory pressure reduces parallelism.

Tested Kernels
1. reduce_occupancy

This kernel allocates a large block of local memory and performs work on it:
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

This kernel stresses local memory usage and forces each work group to reserve approximately 45 KB of shared memory. On GPUs with smaller shared memory budgets per SM, this reduces the number of simultaneously resident work groups, producing measurable occupancy loss. The benchmark helps visualize how severe this effect is.

2. increase_occupancy

This kernel performs the same arithmetic loop but avoids using local memory:
```c
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

Since it uses no shared memory and performs purely ALU work, it typically allows the GPU scheduler to run many more blocks in parallel. This kernel serves as the baseline for maximum occupancy.

Benchmark Methodology

The Python script performs the following steps for each local work group size:

Start with a minimal global size equal to the largest local size.

Increase the global size in steps proportional to the local work group size to keep block counts growing smoothly.

Allocate buffers and enqueue the kernel.

Measure GPU execution time using CommandQueue profiling.

Repeat for all local sizes and record results.

Global sizes sweep from small counts (few blocks) to large counts (many blocks). This exposes how each kernel saturates the GPU and where performance plateaus or drops.

Example code excerpt:

```python
local_sizes = np.array([1, 2, 4, 8, 16, 32, 64])
global_size_arr = np.zeros((len(local_sizes), task_size), dtype=int)
elapsed_times = np.zeros((len(local_sizes), task_size), dtype=float)
```

Each local size produces a full curve that is then plotted on the same graph for comparison.

What You Learn From This Benchmark

How shared memory allocation affects occupancy and prevents multiple blocks from co-existing on a single SM.

How small local sizes reduce SIMD parallelism and often hurt throughput.

How large local sizes may saturate execution units but can also exceed per SM limits depending on the GPU model.

Where the GPU reaches full occupancy and why execution time stops scaling after that point.

How global size interacts with block size to match or mismatch the GPU hardware execution model.

How to Run

Ensure you have:

OpenCL drivers installed

Python 3

PyOpenCL

NumPy

Matplotlib
