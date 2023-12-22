#ifndef __GPU_H
#define __GPU_H

#include <stdint.h>
#include "openclwrapper.h"
#include "string_view.h"

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


#endif
