#ifndef __GPU_H
#define __GPU_H

#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <CL/cl.h>
#include <string.h>
#include <errno.h>

#include "string_builder.h"
#include "constants.h"
#include "logging.h"

#ifdef PROFILING
#define gpu_cl_enqueue_nd(gpu, kernel, n_dim, local, global, offset) gpu_cl_enqueue_nd_profiling(gpu, kernel, n_dim, local, global, offset)
#else
#define gpu_cl_enqueue_nd(gpu, kernel, n_dim, local, global, offset) gpu_cl_enqueue_nd_no_profiling(gpu, kernel, n_dim, local, global, offset)
#endif

#define gpu_cl_create_gpu(gpu, size, flags) gpu_cl_create_gpu_base(gpu, size, flags, __FILE__, __LINE__)
#define gpu_cl_release_memory(mem) gpu_cl_release_memory_base(mem, #mem, __FILE__, __LINE__);

#define gpu_cl_read_gpu(gpu, size, offset, host, device) gpu_cl_read_gpu_base(gpu, size, offset, host, device, #device " -> " #host, __FILE__, __LINE__)
#define gpu_cl_write_gpu(gpu, size, offset, host, device) gpu_cl_write_gpu_base(gpu, size, offset, host, device, #device " <- " #host, __FILE__, __LINE__)


extern uint64_t p_id;
extern uint64_t d_id;
extern uint64_t gpu_optimal_wg;

typedef struct {
    cl_kernel kernel;
    const char *name;
} kernel_t;

typedef struct {
    cl_platform_id *platforms;
    uint64_t n_platforms;

    cl_device_id *devices;
    uint64_t n_devices;

    cl_context ctx;
    cl_command_queue queue;

    cl_program program;
    //Store kernels here?
    kernel_t *kernels;
    uint64_t n_kernels;
} gpu_cl;

gpu_cl gpu_cl_init(const char *current_function, const char *field_func, const char *temperature_func, const char *kernel_augment, const char *compile_augment);
void gpu_cl_close(gpu_cl *gpu);
uint64_t gpu_cl_append_kernel(gpu_cl *gpu, const char *kernel);
void gpu_cl_fill_kernel_args(gpu_cl *gpu, uint64_t kernel, uint64_t offset, uint64_t nargs, ...);
void gpu_cl_enqueue_nd_profiling(gpu_cl *gpu, uint64_t kernel, uint64_t n_dim, uint64_t *local, uint64_t *global, uint64_t *offset);
void gpu_cl_enqueue_nd_no_profiling(gpu_cl *gpu, uint64_t kernel, uint64_t n_dim, uint64_t *local, uint64_t *global, uint64_t *offset);
const char *gpu_cl_get_string_error(cl_int err);
cl_mem gpu_cl_create_gpu_base(gpu_cl *gpu, uint64_t size, cl_mem_flags flags, const char *file, int line);

void gpu_cl_write_gpu_base(gpu_cl *gpu, uint64_t size, uint64_t offset, void *host, cl_mem device, const char *name, const char *file, int line);
void gpu_cl_read_gpu_base(gpu_cl *gpu, uint64_t size, uint64_t offset, void *host, cl_mem device, const char *name, const char *file, int line);

void gpu_cl_set_kernel_arg(gpu_cl *gpu, uint64_t kernel, uint64_t index, uint64_t size, void *data);
void gpu_cl_release_memory_base(cl_mem mem, const char *name, const char *file, int line);
uint64_t gpu_cl_gcd(uint64_t a, uint64_t b);

#endif
