#include <stdint.h>
#include <assert.h>

#include "gpu.h"
#define OPENCLWRAPPER_IMPLEMENTATION
#include "openclwrapper.h"
#include "constants.h"
#include "kernel_funcs.h"
static_assert(sizeof(cl_char4) == sizeof(uint32_t), "Size of cl_char4 is not the same as the size of uint32_t, which should not happen");

uint64_t p_id = 0;
uint64_t d_id = 0;

INCEPTION("Compile OPT is assumed to be storing a null terminated string")
static void gpu_cl_compile_source(gpu_cl *gpu, string_view source, string_view compile_opt) {
    cl_int err;
    gpu->program = clCreateProgramWithSource(gpu->ctx, 1, (const char**)&source.str, &source.len, &err);
    if (err != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not create program on GPU %d: %s", err, clw_get_error_string(err));

    logging_log(LOG_INFO, "Compile OpenCL program with: %.*s\n", (int)compile_opt.len, compile_opt.str);

    cl_int err_building = clBuildProgram(gpu->program, gpu->n_devices, gpu->devices, compile_opt.str, NULL, NULL);

    uint64_t size;
    if ((err = clGetProgramBuildInfo(gpu->program, gpu->devices[d_id], CL_PROGRAM_BUILD_LOG, 0, NULL, &size)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not get program building log size %d: %s", err, clw_get_error_string(err));

    char *info = (char*)calloc(size, 1);
    if (!info)
        logging_log(LOG_FATAL, "Could not calloc buffer for program bulding info");

    if ((err = clGetProgramBuildInfo(gpu->program, gpu->devices[d_id], CL_PROGRAM_BUILD_LOG, size, info, NULL)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not get program building info %d: %s", err, clw_get_error_string(err));

    logging_log(LOG_INFO, "Build Log: \n%s\n", info);
    free(info);

    if (err_building != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not build the program on GPU %d: %s", err_building, clw_get_error_string(err_building));
}

static cl_platform_id *gpu_cl_get_platforms(uint64_t *n) {
    //disable cache
    #if defined(unix) || defined(__unix) || defined(__unix)
    setenv("CUDA_CACHE_DISABLE", "1", 1);
    #elif defined(_WIN32) || defined(_WIN64) || defined(__CYGWIN__)
    _putenv_s("CUDA_CACHE_DISABLE", "1");
    #elif defined(__APPLE__) || defined(__MACH__)
    setenv("CUDA_CACHE_DISABLE", "1", 1);
    #endif

    cl_uint nn;
    cl_int err = clGetPlatformIDs(0, NULL, &nn);
    if (err != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not find number of platforms %d: %s", err, clw_get_error_string(err));
    *n = nn;

    cl_platform_id *local = calloc(sizeof(cl_platform_id) * nn, 1);
    if (!local)
        logging_log(LOG_FATAL, "Could not allocate for platform ids");

    err = clGetPlatformIDs(nn, local, NULL);
    if (err != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not init %u platforms %d: %s", (unsigned int)nn, err, clw_get_error_string(err));
    return local;
}

static void gpu_cl_get_platform_info(cl_platform_id plat, uint64_t iplat) {
    uint64_t n;
    char *info = NULL;
    cl_int err = clGetPlatformInfo(plat, CL_PLATFORM_NAME, 0, NULL, &n);

    if (err != CL_SUCCESS) {
        logging_log(LOG_ERROR, "Could not get platform [%u] name size %d: %s", iplat, err, clw_get_error_string(err));
        goto defer;
    }

    info = calloc(n, 1);
    if (!info) {
        logging_log(LOG_ERROR, "Could not calloc buffer for platform [%u] name", iplat);
        goto defer;
    }

    if ((err = clGetPlatformInfo(plat, CL_PLATFORM_NAME, n, info, NULL)) != CL_SUCCESS) {
        logging_log(LOG_ERROR, "Could not get platform [%u] name %d: %s", iplat, err, clw_get_error_string(err));
        goto defer;
    }

    logging_log(LOG_INFO, "Platform[%zu] name: %s", iplat, info);

defer:
    free(info);
}

static cl_device_id *gpu_cl_get_devices(cl_platform_id plat, uint64_t *n) {
    cl_uint nn;
    cl_int err;
    if ((err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, NULL, &nn)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not find number of devices %d: %s", err, clw_get_error_string(err));
    *n = nn;

    cl_device_id *local = calloc(sizeof(cl_device_id) * nn, 1);
    if (!local)
        logging_log(LOG_FATAL, "Could not calloc for store devices");
    if ((err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, nn, local, NULL)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not initialize devices");
    return local;
}

static void gpu_cl_get_device_info(cl_device_id dev, uint64_t idev) {
    uint64_t n;
    cl_platform_id plt;
    cl_int err;
    char *info = NULL;

    if ((err = clGetDeviceInfo(dev, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &plt, NULL)) != CL_SUCCESS)
        logging_log(LOG_ERROR, "Could not get Platform from Device[%u]", idev);
    else {
        if ((err = clGetPlatformInfo(plt, CL_PLATFORM_NAME, 0, NULL, &n)) != CL_SUCCESS)
            logging_log(LOG_ERROR, "Could not get Platform name from Device[%u]", idev);
        else {
            char *info = calloc(n, 1);
            if (!info)
                logging_log(LOG_FATAL, "Could not calloc for platform name");

            if ((err = clGetPlatformInfo(plt, CL_PLATFORM_NAME, n, info, NULL)) != CL_SUCCESS)
                logging_log(LOG_ERROR, "Could not get platform name for device[%u]", idev);
            else
                logging_log(LOG_INFO, "Device[%u] on Platform %s", idev, info);

            free(info);
            info = NULL;
        }
    }



    err = clGetDeviceInfo(dev, CL_DEVICE_VENDOR, 0, NULL, &n);
    clw_print_cl_error(stderr, err, "ERROR GETTING SIZE DEVICE[%zu] VENDOR INFO", idev);
    info = (char*)CLW_ALLOC(n);
    err = clGetDeviceInfo(dev, CL_DEVICE_VENDOR, n, info, NULL);
    clw_print_cl_error(stderr, err, "ERROR GETTING DEVICE[%zu] VENDOR INFO", idev);
    logging_log(LOG_INFO, "DEVICE[%zu] VENDOR: %s", idev, info);
    free(info);

    err = clGetDeviceInfo(dev, CL_DEVICE_VERSION, 0, NULL, &n);
    clw_print_cl_error(stderr, err, "ERROR GETTING SIZE DEVICE[%zu] VERSION INFO", idev);
    info = (char*)CLW_ALLOC(n);
    err = clGetDeviceInfo(dev, CL_DEVICE_VERSION, n, info, NULL);
    clw_print_cl_error(stderr, err, "ERROR GETTING DEVICE[%zu] VERSION INFO", idev);
    logging_log(LOG_INFO, "DEVICE[%zu] VERSION: %s", idev, info);
    free(info);

    err = clGetDeviceInfo(dev, CL_DRIVER_VERSION, 0, NULL, &n);
    clw_print_cl_error(stderr, err, "ERROR GETTING SIZE DEVICE[%zu] DRIVER VERSION INFO", idev);
    info = (char*)CLW_ALLOC(n);
    err = clGetDeviceInfo(dev, CL_DRIVER_VERSION, n, info, NULL);
    clw_print_cl_error(stderr, err, "ERROR GETTING DEVICE[%zu] DRIVER VERSION INFO", idev);
    logging_log(LOG_INFO, "DEVICE[%zu] DRIVER VERSION: %s", idev, info);
    free(info);

    err = clGetDeviceInfo(dev, CL_DEVICE_NAME, 0, NULL, &n);
    clw_print_cl_error(stderr, err, "ERROR GETTING SIZE DEVICE[%zu] NAME INFO", idev);
    info = (char*)CLW_ALLOC(n);
    err = clGetDeviceInfo(dev, CL_DEVICE_NAME, n, info, NULL);
    clw_print_cl_error(stderr, err, "ERROR GETTING DEVICE[%zu] NAME INFO", idev);
    logging_log(LOG_INFO, "DEVICE[%zu] NAME: %s", idev, info);

    cl_bool device_avaiable;
    err = clGetDeviceInfo(dev, CL_DEVICE_AVAILABLE, sizeof(cl_bool), &device_avaiable, NULL);
    logging_log(LOG_INFO, "DEVICE[%zu] AVAILABLE: %d", idev, device_avaiable);

    cl_ulong memsize;
    err = clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &memsize, NULL);
    clw_print_cl_error(stderr, err, "ERROR GETTING DEVICE[%zu] MEM INFO", idev);
    logging_log(LOG_INFO, "DEVICE[%zu] GLOBAL MEM: %.4f MB", idev, memsize / (1e6));


    cl_ulong meCLW_ALLOC;
    err = clGetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &meCLW_ALLOC, NULL);
    clw_print_cl_error(stderr, err, "ERROR GETTING DEVICE[%zu] MEM INFO", idev);
    logging_log(LOG_INFO, "DEVICE[%zu] ALLOCATABLE MEM: %.4f MB", idev, meCLW_ALLOC / (1e6));

    cl_uint maxcomp;
    err = clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &maxcomp, NULL);
    clw_print_cl_error(stderr, err, "ERROR GETTING DEVICE[%zu] COMPUTE UNITS INFO", idev);
    logging_log(LOG_INFO, "DEVICE[%zu] COMPUTE UNITS: %u", idev, maxcomp);

    uint64_t maxworgroup;
    err = clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(uint64_t), &maxworgroup, NULL);
    clw_print_cl_error(stderr, err, "ERROR GETTING DEVICE[%zu] WORK GROUP INFO", idev);
    logging_log(LOG_INFO, "DEVICE[%zu] MAX WORK GROUP SIZE: %zu", idev, maxworgroup);

    cl_uint dimension;
    err = clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &dimension, NULL);
    clw_print_cl_error(stderr, err, "ERROR GETTING DEVICE[%zu] WORK GROUP INFO", idev);
    logging_log(LOG_INFO, "DEVICE[%zu] MAX DIMENSIONS: %u", idev, dimension);

    uint64_t *dim_size = (uint64_t*)CLW_ALLOC(sizeof(uint64_t) * dimension);
    err = clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(uint64_t) * dimension, dim_size, NULL);
    clw_print_cl_error(stderr, err, "ERROR GETTING DEVICE[%zu] WORK GROUP PER DIMENSION", idev);

    logging_log(LOG_INFO, "DEVICE[%zu] MAX WORK GROUP SIZE PER DIMENSION: {", idev);

    for (uint64_t i = 0; i < dimension - 1; ++i) {
        logging_log(LOG_INFO, "%zu, ", dim_size[i]);
    }
    uint64_t i = dimension - 1;
    logging_log(LOG_INFO, "%zu}", dim_size[i]);
    free(dim_size);

defer:
    free(info);
}

gpu_cl gpu_cl_init(string_view current_function, string_view field_function, string_view temperature_function, string_view kernel_augment, string_view compile_augment) {
    gpu_cl ret = {0};
    ret.platforms = gpu_cl_get_platforms(&ret.n_platforms);
    p_id = p_id % ret.n_platforms;

    for (uint64_t i = 0; i < ret.n_platforms; ++i)
        gpu_cl_get_platform_info(ret.platforms[i], i);

    ret.devices = gpu_cl_get_devices(ret.platforms[p_id], &ret.n_devices);
    d_id = d_id % ret.n_devices;
    for (uint64_t i = 0; i < ret.n_devices; ++i)
        gpu_cl_get_device_info(ret.devices[i], i);

    ret.ctx = clw_init_context(ret.devices, ret.n_devices);
#ifndef PROFILING
    ret.queue = clw_init_queue(ret.ctx, ret.devices[d_id]);
#else
    cl_int err;
    cl_queue_properties properties[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
        0
    };

    ret.queue = clCreateCommandQueueWithProperties(ret.ctx, ret.devices[d_id], properties, &err);
    clw_print_cl_error(stderr, err, "[ FATAL ] Could not create command queue");
#endif

    const char cmp[] = "-DOPENCL_COMPILATION";
    string kernel = fill_functions_on_kernel(current_function, field_function, temperature_function, kernel_augment);
    string compile = fill_compilation_params(sv_from_cstr(cmp), compile_augment);
    gpu_cl_compile_source(&ret, sv_from_cstr(string_as_cstr(&kernel)), sv_from_cstr(string_as_cstr(&compile)));
    string_free(&kernel);
    string_free(&compile);

    return ret;
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

/*uint64_t gpu_profiling_base(FILE *f, cl_event ev, const char *description) {
    uint64_t start, end, duration;
    uint64_t size;
    clw_print_cl_error(stderr, clWaitForEvents(1, &ev), "[ FATAL ] Could not wait for event \"%s\"", description);
    clw_print_cl_error(stderr, clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(start), &start, &size), "[ FATAL ] Could not retrieve submit information from profiling \"%s\"", description);
    clw_print_cl_error(stderr, clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(end), &end, &size), "[ FATAL ] Could not retrieve complete information from profiling \"%s\"", description);
    duration = end - start;
    fprintf(f, "%s Spent %e us\n", description, (double)duration / 1000.0);
    //clReleaseEvent(ev);
    return duration;
}*/
