#ifndef __GPU_H
#define __GPU_H

#include <stdint.h>
#include "openclwrapper.h"
#include "string_view.h"
#include <stdarg.h>

/*#define NUMARGS(...) (sizeof((int[]){__VA_ARGS__})/sizeof(int))
#define gpu_fill_kernel_args(gpu, kernel, offset, ...) (gpu_fill_kernel_args_base(gpu, kernel, offset, NUMARGS(__VA_ARGS__), __VA_ARGS__))*/

typedef struct {
    cl_platform_id *platforms;
    uint64_t n_platforms;
    int plat_idx;

    cl_device_id *devices;
    uint64_t n_devices;
    int dev_idx;

    cl_context ctx;
    cl_command_queue queue;

    cl_program program;
    //Store kernels here?
    kernel_t *kernels;
    uint64_t n_kernels;
} gpu_cl;

gpu_cl gpu_cl_init(int plat_idx, int dev_idx);
void gpu_cl_compile_source(gpu_cl *gpu, string_view source, string_view compile_opt);
void gpu_cl_close(gpu_cl *gpu);
uint64_t gpu_append_kernel(gpu_cl *gpu, const char *kernel);
void gpu_fill_kernel_args(gpu_cl *gpu, uint64_t kernel, uint64_t offset, uint64_t nargs, ...);


#endif
