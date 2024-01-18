#include <stdint.h>
#include <inttypes.h>
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

    char *info = calloc(size, 1);
    if (!info)
        logging_log(LOG_FATAL, "Could not calloc[%u bytes] buffer for program bulding info: %s", size, strerror(errno));

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
        logging_log(LOG_FATAL, "Could not allocate[%"PRIu64" bytes] for platform ids: %s", sizeof(cl_platform_id) * nn, strerror(errno));

    err = clGetPlatformIDs(nn, local, NULL);
    if (err != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not init %"PRIu64" platforms %d: %s", (unsigned int)nn, err, clw_get_error_string(err));
    return local;
}

static void gpu_cl_get_platform_info(cl_platform_id plat, uint64_t iplat) {
    uint64_t n;
    char *info = NULL;
    cl_int err = clGetPlatformInfo(plat, CL_PLATFORM_NAME, 0, NULL, &n);

    if (err != CL_SUCCESS) {
        logging_log(LOG_ERROR, "Could not get platform [%"PRIu64"] name size %d: %s", iplat, err, clw_get_error_string(err));
        goto defer;
    }

    info = calloc(n, 1);
    if (!info) {
        logging_log(LOG_ERROR, "Could not calloc[%"PRIu64" bytes] buffer for platform ["PRIu64"] name: %s", n, iplat, strerror(errno));
        goto defer;
    }

    if ((err = clGetPlatformInfo(plat, CL_PLATFORM_NAME, n, info, NULL)) != CL_SUCCESS) {
        logging_log(LOG_ERROR, "Could not get platform [%"PRIu64"] name %d: %s", iplat, err, clw_get_error_string(err));
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
        logging_log(LOG_FATAL, "Could not calloc[%"PRIu64" bytes] for store devices: %s", sizeof(cl_device_id) * nn, strerror(errno));
    if ((err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, nn, local, NULL)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not initialize devices %d: %s", err, clw_get_error_string(err));
    return local;
}

static void gpu_cl_get_device_info(cl_device_id dev, uint64_t idev) {
    uint64_t n;
    cl_platform_id plt;
    cl_int err;
    char *info = NULL;

    if ((err = clGetDeviceInfo(dev, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &plt, NULL)) != CL_SUCCESS)
        logging_log(LOG_ERROR, "Could not get Platform from Device[%"PRIu64"] %d: %s", idev, err, clw_get_error_string(err));
    else {
        if ((err = clGetPlatformInfo(plt, CL_PLATFORM_NAME, 0, NULL, &n)) != CL_SUCCESS)
            logging_log(LOG_ERROR, "Could not get Platform name from Device[%"PRIu64"] %d: %s", idev, err, clw_get_error_string(err));
        else {
            char *info = calloc(n, 1);
            if (!info)
                logging_log(LOG_FATAL, "Could not calloc[%"PRIu64" bytes] for platform name: %s", n, strerror(errno));

            if ((err = clGetPlatformInfo(plt, CL_PLATFORM_NAME, n, info, NULL)) != CL_SUCCESS)
                logging_log(LOG_ERROR, "Could not get platform name for device[%"PRIu64"] %d: %s", idev, err, clw_get_error_string(err));
            else
                logging_log(LOG_INFO, "Device[%"PRIu64"] on Platform %s", idev, info);

            free(info);
            info = NULL;
        }
    }

    if ((err = clGetDeviceInfo(dev, CL_DEVICE_VENDOR, 0, NULL, &n)) != CL_SUCCESS)
        logging_log(LOG_ERROR, "Could not get Device[%"PRIu64"] Vendor Size %d: %s", idev, err, clw_get_error_string(err));
    else {
        info = calloc(n, 1);
        if (!info)
            logging_log(LOG_ERROR, "Could not calloc[%"PRIu64" bytes] for Device[%"PRIu64"] Vendor: %s", n, idev, strerror(errno));
        else {
            if ((err = clGetDeviceInfo(dev, CL_DEVICE_VENDOR, n, info, NULL)) != CL_SUCCESS)
                logging_log(LOG_ERROR, "Could not get Device[%"PRIu64"] Vendor info %d: %s", idev, err, clw_get_error_string(err));
            else
                logging_log(LOG_INFO, "Device[%"PRIu64"] Vendor: %s", idev, info);
        }
        free(info);
    }

    if ((err = clGetDeviceInfo(dev, CL_DEVICE_VERSION, 0, NULL, &n)) != CL_SUCCESS)
        logging_log(LOG_ERROR, "Could not get Device[%"PRIu64"] Version %d: %s", idev, err, clw_get_error_string(err));
    else {
        info = calloc(n, 1);
        if (!info)
            logging_log(LOG_ERROR, "Could not calloc[%"PRIu64" bytes] for Device[%"PRIu64"] Version: %s", n, idev, strerror(errno));
        else {
            if ((err = clGetDeviceInfo(dev, CL_DEVICE_VERSION, n, info, NULL)) != CL_SUCCESS)
                logging_log(LOG_ERROR, "Could not get Device[%"PRIu64"] Version %d: %s", idev, err, clw_get_error_string(err));
            else
                logging_log(LOG_INFO, "Device[%"PRIu64"] Version: %s", idev, info);
        }
        free(info);
    }

    if ((err = clGetDeviceInfo(dev, CL_DRIVER_VERSION, 0, NULL, &n)) != CL_SUCCESS)
        logging_log(LOG_ERROR, "Could not get Device[%"PRIu64"] Driver Version %d: %s", idev, err, clw_get_error_string(err));
    else {
        info = calloc(n, 1);
        if (!info)
            logging_log(LOG_ERROR, "Could not calloc[%"PRIu64" bytes] for Device[%"PRIu64"] Driver Version: %s", n, idev, strerror(errno));
        else {
            if ((err = clGetDeviceInfo(dev, CL_DRIVER_VERSION, n, info, NULL)) != CL_SUCCESS)
                logging_log(LOG_ERROR, "Could not get Device[%"PRIu64"] Driver Version %d: %s", idev, err, clw_get_error_string(err));
            else
                logging_log(LOG_INFO, "Device[%"PRIu64"] Driver Version: %s", idev, info);
        }
        free(info);
    }

    if ((err = clGetDeviceInfo(dev, CL_DEVICE_NAME, 0, NULL, &n)) != CL_SUCCESS)
        logging_log(LOG_ERROR, "Could not get Device[%"PRIu64"] Name %d: %s", idev, err, clw_get_error_string(err));
    else {
        info = calloc(n, 1);
        if (!info)
            logging_log(LOG_ERROR, "Could not calloc[%"PRIu64" bytes] for Device[%"PRIu64"] Name: %s", n, idev, strerror(errno));
        else {
            if ((err = clGetDeviceInfo(dev, CL_DEVICE_NAME, n, info, NULL)) != CL_SUCCESS)
                logging_log(LOG_ERROR, "Could not get Device[%"PRIu64"] Name %d: %s", idev, err, clw_get_error_string(err));
            else
                logging_log(LOG_INFO, "Device[%"PRIu64"] Name: %s", idev, info);
        }
        free(info);
    }

    cl_bool device_avaiable;
    if ((err = clGetDeviceInfo(dev, CL_DEVICE_AVAILABLE, sizeof(cl_bool), &device_avaiable, NULL)) != CL_SUCCESS)
        logging_log(LOG_ERROR, "Could not get if Device[%"PRIu64"] is available %d: %s", idev, err, clw_get_error_string(err));
    else
        logging_log(LOG_INFO, "Device[%"PRIu64"] Available: %d", idev, device_avaiable);

    cl_ulong memsize;
    if ((err = clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &memsize, NULL)) != CL_SUCCESS)
        logging_log(LOG_ERROR, "Could not get Device[%"PRIu64"] memory size %d: %s", idev, err, clw_get_error_string(err));
    else
        logging_log(LOG_INFO, "Device[%"PRIu64"] Global Memory: %.4f MB", idev, memsize / (1.0e6));


    cl_ulong mem_allocable;
    if ((err = clGetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &mem_allocable, NULL)) != CL_SUCCESS)
        logging_log(LOG_ERROR, "Could not get Device[%"PRIu64"] allocable memory %d: %s", idev, err, clw_get_error_string(err));
    else
        logging_log(LOG_INFO, "Device[%"PRIu64"] Allocatable Memory: %.4f MB", idev, mem_allocable / (1.0e6));

    cl_uint maxcomp;
    if ((err = clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &maxcomp, NULL)) != CL_SUCCESS)
        logging_log(LOG_ERROR, "Could not get Device[%"PRIu64"] max compute units %d: %s", idev, err, clw_get_error_string(err));
    else
        logging_log(LOG_INFO, "Device[%"PRIu64"] Compute Units: %"PRIu64"", idev, maxcomp);

    uint64_t maxworgroup;
    if ((err = clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(uint64_t), &maxworgroup, NULL)) != CL_SUCCESS)
        logging_log(LOG_ERROR, "Could not get Device[%"PRIu64"] work group size %d: %s", idev, err, clw_get_error_string(err));
    else
        logging_log(LOG_INFO, "Device[%"PRIu64"] Max Work Group Size: %"PRIu64"", idev, maxworgroup);

    cl_uint dimension;
    if ((err = clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &dimension, NULL)) != CL_SUCCESS)
        logging_log(LOG_ERROR, "Could not get Device[%"PRIu64"] max work item dimensions %d: %s", idev, err, clw_get_error_string(err));
    else {
        logging_log(LOG_INFO, "Device[%"PRIu64"] Max Dimension: %"PRIu64"", idev, dimension);

        uint64_t dim_size[dimension];
        if ((err = clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(uint64_t) * dimension, dim_size, NULL)) != CL_SUCCESS)
            logging_log(LOG_ERROR, "Could not get Device[%"PRIu64"] max work items per dimension %d: %s", idev, err, clw_get_error_string(err));
        else {
            char buffer[1024] = {0};
            char *ptr = buffer;
            int adv = 0;

            for (uint64_t i = 0; i < dimension - 1; ++i)
                adv += snprintf(ptr + adv, 1023 - adv, "%"PRIu64", ", dim_size[i]);

            uint64_t i = dimension - 1;
            adv += snprintf(ptr + adv, 1023 - adv, "%"PRIu64"", dim_size[i]);

            logging_log(LOG_INFO, "Device[%"PRIu64"] Max Work Items Per Dimension: {%s]", idev, buffer);
        }
    }
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
