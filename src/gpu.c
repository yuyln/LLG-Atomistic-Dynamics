#include <stdint.h>
#include <assert.h>

#include "gpu.h"
#define OPENCLWRAPPER_IMPLEMENTATION
#include "openclwrapper.h"
#include "constants.h"
static_assert(sizeof(cl_char4) == sizeof(uint32_t), "Size of cl_char4 is not the same as the size of uint32_t, which should not happen");

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
#ifndef PROFILING
    ret.queue = clw_init_queue(ret.ctx, ret.devices[ret.dev_idx]);
#else
    cl_int err;
    cl_queue_properties properties[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
        0
    };

    ret.queue = clCreateCommandQueueWithProperties(ret.ctx, ret.devices[ret.dev_idx], properties, &err);
    clw_print_cl_error(stderr, err, "[ FATAL ] Could not create command queue");
#endif
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

uint64_t gpu_append_kernel(gpu_cl *gpu, const char *kernel) {
    cl_int err;
    gpu->kernels = realloc(gpu->kernels, sizeof(*gpu->kernels) * (gpu->n_kernels + 1));
    gpu->kernels[gpu->n_kernels].name = kernel;
    gpu->kernels[gpu->n_kernels].kernel = clCreateKernel(gpu->program, kernel, &err);
    clw_print_cl_error(stderr, err, "[ FATAL ] Could not load kernel %s", kernel);
    uint64_t index = gpu->n_kernels;
    ++gpu->n_kernels;
    return index;
}

void gpu_fill_kernel_args(gpu_cl *gpu, uint64_t kernel, uint64_t offset, uint64_t nargs, ...) {
    va_list arg_list;
    va_start(arg_list, nargs);
    for (uint64_t i = offset; i < offset + nargs; ++i) {
        void *item = (void*)va_arg(arg_list,  uint64_t);
        uint64_t sz = va_arg(arg_list, uint64_t);
        clw_print_cl_error(stderr, clSetKernelArg(gpu->kernels[kernel].kernel, i, sz, item), "[ FATAL ] Could not set argument %d of kernel %s", (int)i, gpu->kernels[kernel].name);
    }
    va_end(arg_list);
}

uint64_t gpu_profiling_base(FILE *f, cl_event ev, const char *description) {
    uint64_t start, end, duration;
    uint64_t size;
    clw_print_cl_error(stderr, clWaitForEvents(1, &ev), "[ FATAL ] Could not wait for event \"%s\"", description);
    clw_print_cl_error(stderr, clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(start), &start, &size), "[ FATAL ] Could not retrieve submit information from profiling \"%s\"", description);
    clw_print_cl_error(stderr, clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(end), &end, &size), "[ FATAL ] Could not retrieve complete information from profiling \"%s\"", description);
    duration = end - start;
    fprintf(f, "%s Spent %e us\n", description, (double)duration / 1000.0);
    return duration;
}
