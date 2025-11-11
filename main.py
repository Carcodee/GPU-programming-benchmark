import pyopencl as cl
from pyopencl.tools import get_test_platforms_and_devices

import numpy as np
import time
import matplotlib.pyplot as plt
print(get_test_platforms_and_devices())
ctx = cl.create_some_context()
PYOPENCL_CTX='0'
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

prg = cl.Program(ctx, """
    __kernel void vector_add(__global float *a, __global float *c)
    {
         int gid = get_global_id(0);
         for (int i = 0; i < 1000000; ++i) {
            a[gid] *= get_local_id(0);
         }
         c[gid] = get_local_id(0);
    }
    """).build()

#params
task_size = 200
local_sizes = np.array([1, 2, 4, 8, 16, 32, 64])
colors = ['blue', 'red', 'magenta', 'green', 'orange', 'black', 'purple', 'cyan', 'coral', 'xkcd:sky blue', 'xkcd:brick red']

global_size_arr = np.zeros((len(local_sizes), task_size), dtype=int)
elapsed_times = np.zeros((len(local_sizes), task_size), dtype=float)

#params
curr_size= local_sizes[-1] 
step_size = 2

task_index=0 
local_size_idx = 0


while local_size_idx < len(local_sizes):

   local_size = (local_sizes[local_size_idx], 1, 1)

   global_size_arr[local_size_idx, task_index] = curr_size
   global_size = (global_size_arr[local_size_idx, task_index], 1, 1)

   a = np.random.rand(global_size[0]).astype(np.float32)
   c = np.zeros(global_size[0], dtype=np.float32)

   mf = cl.mem_flags
   a_buf = cl.Buffer\
      (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
   c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)

   start = time.time()
   event = prg.vector_add(queue, global_size, local_size, a_buf, c_buf) 
   event.wait()  # block until GPU finishes
   end = time.time()

   elapsed_times[local_size_idx, task_index] = (end - start) * 1000

   print(f'{task_index}: Global size: {global_size[0]}, local size: {local_size}, Time: {elapsed_times[local_size_idx, task_index]:.2f} MS,')

   #c buffer in host to copy
   c_result = np.empty_like(c)
   cl.enqueue_copy(queue, c_result, c_buf)

   task_index+= 1
   curr_size += step_size * local_sizes[local_size_idx]
   if task_index >= task_size:
      task_index = 0
      local_size_idx+= 1
      curr_size = local_sizes[-1]

print(c_result)
plt.close('all')
plt.figure(figsize=(10, 5))
for i in range(len(local_sizes)):
   plt.plot(global_size_arr[i].astype(float), elapsed_times[i], label = f"local size= {local_sizes[i]}", linestyle = '-', color = colors[i])

plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.ticklabel_format(style = 'plain')
plt.tight_layout()
plt.xlabel("Global Work Size")
plt.ylabel("Execution Time (ms)")
plt.show()
