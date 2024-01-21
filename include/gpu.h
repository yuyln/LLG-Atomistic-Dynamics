#ifndef __GPU_H
#define __GPU_H

#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>

#include "openclwrapper.h"
#include "string_view.h"
#include "constants.h"
#include "logging.h"

#ifdef PROFILING
#define gpu_cl_enqueue_nd
#else
#define gpu_cl_enqueue_nd
#endif


extern uint64_t p_id;
extern uint64_t d_id;

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
uint64_t gpu_append_kernel(gpu_cl *gpu, const char *kernel);
void gpu_fill_kernel_args(gpu_cl *gpu, uint64_t kernel, uint64_t offset, uint64_t nargs, ...);


#endif
