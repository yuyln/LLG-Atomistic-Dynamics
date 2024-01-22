#ifndef __GPU_H
#define __GPU_H

#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <CL/cl.h>
#include <string.h>
#include <errno.h>

#include "string_view.h"
#include "constants.h"
#include "logging.h"

#ifdef PROFILING
#define gpu_cl_enqueue_nd(gpu, kernel, n_dim, local, global, offset) gpu_cl_enqueue_nd_profiling(gpu, kernel, n_dim, local, global, offset)
#else
#define gpu_cl_enqueue_nd(gpu, kernel, n_dim, local, global, offset) gpu_cl_enqueue_nd_no_profiling(gpu, kernel, n_dim, local, global, offset)
#endif

#define gpu_cl_create_buffer(gpu, size, flags) gpu_cl_create_buffer_base(gpu, size, flags, __FILE__, __LINE__)


extern uint64_t p_id;
extern uint64_t d_id;

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

gpu_cl gpu_cl_init(string_view current_function, string_view field_function, string_view temperature_function, string_view kernel_augment, string_view compile_augment);
void gpu_cl_close(gpu_cl *gpu);
uint64_t gpu_cl_append_kernel(gpu_cl *gpu, const char *kernel);
void gpu_cl_fill_kernel_args(gpu_cl *gpu, uint64_t kernel, uint64_t offset, uint64_t nargs, ...);
void gpu_cl_enqueue_nd_profiling(gpu_cl *gpu, uint64_t kernel, uint64_t n_dim, uint64_t *local, uint64_t *global, uint64_t *offset);
void gpu_cl_enqueue_nd_no_profiling(gpu_cl *gpu, uint64_t kernel, uint64_t n_dim, uint64_t *local, uint64_t *global, uint64_t *offset);
const char *gpu_cl_get_string_error(cl_int err);
cl_mem gpu_cl_create_buffer_base(gpu_cl *gpu, uint64_t size, cl_mem_flags flags, const char *file, int line);
void gpu_cl_write_buffer(gpu_cl *gpu, uint64_t size, uint64_t offset, void *host, cl_mem device);
void gpu_cl_read_buffer(gpu_cl *gpu, uint64_t size, uint64_t offset, void *host, cl_mem device);
void gpu_cl_set_kernel_arg(gpu_cl *gpu, uint64_t kernel, uint64_t index, uint64_t size, void *data);
uint64_t gpu_cl_gcd(uint64_t a, uint64_t b);

#endif
