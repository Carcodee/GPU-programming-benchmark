GPU-programming-benchmark

this is a small performance test where I compare multiple cases, with different local work group size and different kernels.
The main purpose of this project was to understand the GPU model and how cores, streaming multiprocesors, blocks and threats behave internally.
The figures under this readme show how a simple kernnel such as 

```c
some code
```

behave with different local_work_group sizes.
