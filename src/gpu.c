#include <stdint.h>

#include "gpu.h"
#define OPENCLWRAPPER_IMPLEMENTATION
#include "openclwrapper.h"
#include "constants.h"

//@TODO: Create queue with properties
gpu_cl gpu_cl_init(int plat_idx, int dev_idx) {
    gpu_cl ret = {0};
    ret.platforms = clw_init_platforms(&ret.n_platforms);
    ret.plat_idx = plat_idx % ret.n_platforms;

    for (uint64_t i = 0; i < ret.n_platforms; ++i)
        clw_get_platform_info(stdout, ret.platforms[i], i);

    ret.devices = clw_init_devices(ret.platforms[ret.plat_idx], &ret.n_devices);
    ret.dev_idx = dev_idx % ret.n_devices;
    for (uint64_t i = 0; i < ret.n_devices; ++i)
        clw_get_device_info(stdout, ret.devices[i], i);

    ret.ctx = clw_init_context(ret.devices, ret.n_devices);
    ret.queue = clw_init_queue(ret.ctx, ret.devices[ret.dev_idx]);
    return ret;
}

INCEPTION("Compile OPT is assumed to be storing a null terminated string")
void gpu_cl_compile_source(gpu_cl *gpu, string_view source, string_view compile_opt) {
    cl_int err;
    gpu->program = clCreateProgramWithSource(gpu->ctx, 1, (const char**)&source.str, &source.len, &err);
    clw_print_cl_error(stderr, err, "[ FATAL ] Creating program");

    printf("[ INFO ] Compile OpenCL Options: %s\n", compile_opt.str);
    cl_int err_building = clw_build_program(gpu->program, gpu->n_devices, gpu->devices, compile_opt.str);

    uint64_t size;
    err = clGetProgramBuildInfo(gpu->program, gpu->devices[gpu->dev_idx], CL_PROGRAM_BUILD_LOG, 0, NULL, &size);
    clw_print_cl_error(stderr, err, "[ FATAL ] Could not get building log size");

    char *info = (char*)calloc(size, 1);
    err = clGetProgramBuildInfo(gpu->program, gpu->devices[gpu->dev_idx], CL_PROGRAM_BUILD_LOG, size, info, NULL);
    clw_print_cl_error(stderr, err, "[ FATAL ] Error getting building log");

    fprintf(stderr, "---------------------------------\n");
    fprintf(stderr, "[ INFO ] BUILD LOG: \n%s\n", info);
    fprintf(stderr, "---------------------------------\n");
    free(info);
    clw_print_cl_error(stderr, err_building, "[ FATAL ] Could not build the program");
}

void gpu_cl_close(gpu_cl *gpu) {
    for (uint64_t i = 0; i < gpu->n_kernels; ++i)
        clw_print_cl_error(stderr, clReleaseKernel(gpu->kernels[i].kernel), "[ FATAL ] Could not release kernel %s", gpu->kernels[i].name);
    free(gpu->kernels);

    clw_print_cl_error(stderr, clReleaseProgram(gpu->program), "[ FATAL ] Could not release program");
    clw_print_cl_error(stderr, clReleaseCommandQueue(gpu->queue), "[ FATAL ] Could not release command queue");
    clw_print_cl_error(stderr, clReleaseContext(gpu->ctx), "[ FATAL ] Could not release context");

    for (uint64_t i = 0; i < gpu->n_devices; ++i)
        clw_print_cl_error(stderr, clReleaseDevice(gpu->devices[i]), "[ FATAL ] Could not release device %zu", i);
    free(gpu->devices);
    free(gpu->platforms);
    memset(gpu, 0, sizeof(*gpu));
}
