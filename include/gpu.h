#ifndef __GPU_H
#define __GPU_H

#include <stdint.h>
#include "openclwrapper.h"

typedef struct {
    cl_platform_id *platforms;
    uint64_t n_platforms;

    cl_device_id *devices;
    uint64_t n_devices;

    cl_context ctx;
    cl_command_queue queue;
    //Store kernels here?
} gpu_data;

#endif
