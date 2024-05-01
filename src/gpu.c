#include <stdint.h>
#include <inttypes.h>
#include <assert.h>

#include "gpu.h"
#include "logging.h"
#include "constants.h"
#include "kernel_funcs.h"
#include "allocator.h"
static_assert(sizeof(cl_char4) == sizeof(uint32_t), "Size of cl_char4 is not the same as the size of uint32_t, which should not happen");

uint64_t p_id = 0;
uint64_t d_id = 0;

static const char *errors[60] = {"CL_SUCCESS",
                                 "CL_DEVICE_NOT_FOUND",
                                 "CL_DEVICE_NOT_AVAILABLE",
                                 "CL_COMPILER_NOT_AVAILABLE",
                                 "CL_MEM_OBJECT_ALLOCATION_FAILURE",
                                 "CL_OUT_OF_RESOURCES",
                                 "CL_OUT_OF_HOST_MEMORY",
                                 "CL_PROFILING_INFO_NOT_AVAILABLE",
                                 "CL_MEM_COPY_OVERLAP",
                                 "CL_IMAGE_FORMAT_MISMATCH",
                                 "CL_IMAGE_FORMAT_NOT_SUPPORTED",
                                 "CL_BUILD_PROGRAM_FAILURE",
                                 "CL_MAP_FAILURE",
                                 "CL_MISALIGNED_SUB_BUFFER_OFFSET",
                                 "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
                                 "CL_COMPILE_PROGRAM_FAILURE",
                                 "CL_LINKER_NOT_AVAILABLE",
                                 "CL_LINK_PROGRAM_FAILURE",
                                 "CL_DEVICE_PARTITION_FAILED",
                                 "CL_KERNEL_ARG_INFO_NOT_AVAILABLE",
                                 "CL_INVALID_VALUE",
                                 "CL_INVALID_DEVICE_TYPE",
                                 "CL_INVALID_PLATFORM",
                                 "CL_INVALID_DEVICE",
                                 "CL_INVALID_CONTEXT",
                                 "CL_INVALID_QUEUE_PROPERTIES",
                                 "CL_INVALID_COMMAND_QUEUE",
                                 "CL_INVALID_HOST_PTR",
                                 "CL_INVALID_MEM_OBJECT",
                                 "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
                                 "CL_INVALID_IMAGE_SIZE",
                                 "CL_INVALID_SAMPLER",
                                 "CL_INVALID_BINARY",
                                 "CL_INVALID_BUILD_OPTIONS",
                                 "CL_INVALID_PROGRAM",
                                 "CL_INVALID_PROGRAM_EXECUTABLE",
                                 "CL_INVALID_KERNEL_NAME",
                                 "CL_INVALID_KERNEL_DEFINITION",
                                 "CL_INVALID_KERNEL",
                                 "CL_INVALID_ARG_INDEX",
                                 "CL_INVALID_ARG_VALUE",
                                 "CL_INVALID_ARG_SIZE",
                                 "CL_INVALID_KERNEL_ARGS",
                                 "CL_INVALID_WORK_DIMENSION",
                                 "CL_INVALID_WORK_GROUP_SIZE",
                                 "CL_INVALID_WORK_ITEM_SIZE",
                                 "CL_INVALID_GLOBAL_OFFSET",
                                 "CL_INVALID_EVENT_WAIT_LIST",
                                 "CL_INVALID_EVENT",
                                 "CL_INVALID_OPERATION",
                                 "CL_INVALID_GL_OBJECT",
                                 "CL_INVALID_BUFFER_SIZE",
                                 "CL_INVALID_MIP_LEVEL",
                                 "CL_INVALID_GLOBAL_WORK_SIZE",
                                 "CL_INVALID_PROPERTY",
                                 "CL_INVALID_IMAGE_DESCRIPTOR",
                                 "CL_INVALID_COMPILER_OPTIONS",
                                 "CL_INVALID_LINKER_OPTIONS",
                                 "CL_INVALID_DEVICE_PARTITION_COUNT"};

const char *gpu_cl_get_str_error(cl_int err) {
    int err_ = abs(err);                                                        
    if (err_ >= 30)                                                               
        err_ = err_ - 10;                                                         
    return errors[err_];
}

INCEPTION("Compile OPT is assumed to be storing a null terminated string")
static void gpu_cl_compile_source(gpu_cl *gpu, string source, string compile_opt) {
    cl_int err;
    gpu->program = clCreateProgramWithSource(gpu->ctx, 1, (const char**)&source.str, &source.len, &err);
    if (err != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not create program on GPU %d: %s", err, gpu_cl_get_str_error(err));

    logging_log(LOG_INFO, "Compile OpenCL program with: %.*s", (int)compile_opt.len, compile_opt.str);

    cl_int err_building = clBuildProgram(gpu->program, gpu->n_devices, gpu->devices, compile_opt.str, NULL, NULL);

    uint64_t size;
    if ((err = clGetProgramBuildInfo(gpu->program, gpu->devices[d_id], CL_PROGRAM_BUILD_LOG, 0, NULL, &size)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not get program building log size %d: %s", err, gpu_cl_get_str_error(err));

    char *info = mmalloc(size);
    if ((err = clGetProgramBuildInfo(gpu->program, gpu->devices[d_id], CL_PROGRAM_BUILD_LOG, size, info, NULL)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not get program building info %d: %s", err, gpu_cl_get_str_error(err));

    logging_log(LOG_INFO, "Build Log: \n%s", info);
    mfree(info);

    if (err_building != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not build the program on GPU %d: %s", err_building, gpu_cl_get_str_error(err_building));
}

void gpu_cl_get_platforms(gpu_cl *gpu) {
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
        logging_log(LOG_FATAL, "Could not find number of platforms %d: %s", err, gpu_cl_get_str_error(err));
    gpu->n_platforms = nn;

    gpu->platforms = mmalloc(sizeof(cl_platform_id) * nn);

    err = clGetPlatformIDs(nn, gpu->platforms, NULL);
    if (err != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not init %"PRIu64" platforms %d: %s", gpu->n_platforms, err, gpu_cl_get_str_error(err));
    logging_log(LOG_INFO, "Initialized %"PRIu64" platforms", gpu->n_platforms);
}

static void gpu_cl_get_platform_info(cl_platform_id plat, uint64_t iplat) {
    uint64_t n;
    char *info = NULL;
    cl_int err = clGetPlatformInfo(plat, CL_PLATFORM_NAME, 0, NULL, &n);

    if (err != CL_SUCCESS) {
        logging_log(LOG_ERROR, "Could not get platform [%"PRIu64"] name size %d: %s", iplat, err, gpu_cl_get_str_error(err));
        goto defer;
    }

    info = mmalloc(n);

    if ((err = clGetPlatformInfo(plat, CL_PLATFORM_NAME, n, info, NULL)) != CL_SUCCESS) {
        logging_log(LOG_ERROR, "Could not get platform [%"PRIu64"] name %d: %s", iplat, err, gpu_cl_get_str_error(err));
        goto defer;
    }

    logging_log(LOG_INFO, "Platform[%zu] name: %s", iplat, info);

defer:
    mfree(info);
}

static void gpu_cl_get_devices(gpu_cl *gpu) {
    cl_uint nn;
    cl_int err;
    if ((err = clGetDeviceIDs(gpu->platforms[p_id], CL_DEVICE_TYPE_ALL, 0, NULL, &nn)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not find number of devices %d: %s", err, gpu_cl_get_str_error(err));
    gpu->n_devices = nn;

    gpu->devices = mmalloc(sizeof(cl_device_id) * nn);
    if ((err = clGetDeviceIDs(gpu->platforms[p_id], CL_DEVICE_TYPE_ALL, nn, gpu->devices, NULL)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not initialize devices %d: %s", err, gpu_cl_get_str_error(err));

    logging_log(LOG_INFO, "Initialized %"PRIu64" devices", gpu->n_devices);
}

static void gpu_cl_get_device_info(cl_device_id dev, uint64_t idev) {
    uint64_t n;
    cl_platform_id plt;
    cl_int err;
    char *info = NULL;

    if ((err = clGetDeviceInfo(dev, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &plt, NULL)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not get Platform from Device[%"PRIu64"] %d: %s", idev, err, gpu_cl_get_str_error(err));

    if ((err = clGetPlatformInfo(plt, CL_PLATFORM_NAME, 0, NULL, &n)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not get Platform name from Device[%"PRIu64"] %d: %s", idev, err, gpu_cl_get_str_error(err));

    info = mmalloc(n + 1);
    if ((err = clGetPlatformInfo(plt, CL_PLATFORM_NAME, n, info, NULL)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not get platform name for device[%"PRIu64"] %d: %s", idev, err, gpu_cl_get_str_error(err));

    logging_log(LOG_INFO, "Device[%"PRIu64"] on Platform %s", idev, info);

    mfree(info);
    info = NULL;

    if ((err = clGetDeviceInfo(dev, CL_DEVICE_VENDOR, 0, NULL, &n)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not get Device[%"PRIu64"] Vendor Size %d: %s", idev, err, gpu_cl_get_str_error(err));

    info = mmalloc(n);
    if ((err = clGetDeviceInfo(dev, CL_DEVICE_VENDOR, n, info, NULL)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not get Device[%"PRIu64"] Vendor info %d: %s", idev, err, gpu_cl_get_str_error(err));

    logging_log(LOG_INFO, "Device[%"PRIu64"] Vendor: %s", idev, info);

    mfree(info);

    if ((err = clGetDeviceInfo(dev, CL_DEVICE_VERSION, 0, NULL, &n)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not get Device[%"PRIu64"] Version %d: %s", idev, err, gpu_cl_get_str_error(err));

    info = mmalloc(n);

    if ((err = clGetDeviceInfo(dev, CL_DEVICE_VERSION, n, info, NULL)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not get Device[%"PRIu64"] Version %d: %s", idev, err, gpu_cl_get_str_error(err));

    logging_log(LOG_INFO, "Device[%"PRIu64"] Version: %s", idev, info);

    mfree(info);

    if ((err = clGetDeviceInfo(dev, CL_DRIVER_VERSION, 0, NULL, &n)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not get Device[%"PRIu64"] Driver Version %d: %s", idev, err, gpu_cl_get_str_error(err));

    info = mmalloc(n);

    if ((err = clGetDeviceInfo(dev, CL_DRIVER_VERSION, n, info, NULL)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not get Device[%"PRIu64"] Driver Version %d: %s", idev, err, gpu_cl_get_str_error(err));

    logging_log(LOG_INFO, "Device[%"PRIu64"] Driver Version: %s", idev, info);

    mfree(info);

    if ((err = clGetDeviceInfo(dev, CL_DEVICE_NAME, 0, NULL, &n)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not get Device[%"PRIu64"] Name %d: %s", idev, err, gpu_cl_get_str_error(err));

    info = mmalloc(n);

    if ((err = clGetDeviceInfo(dev, CL_DEVICE_NAME, n, info, NULL)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not get Device[%"PRIu64"] Name %d: %s", idev, err, gpu_cl_get_str_error(err));

    logging_log(LOG_INFO, "Device[%"PRIu64"] Name: %s", idev, info);

    mfree(info);

    cl_bool device_avaiable;
    if ((err = clGetDeviceInfo(dev, CL_DEVICE_AVAILABLE, sizeof(cl_bool), &device_avaiable, NULL)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not get if Device[%"PRIu64"] is available %d: %s", idev, err, gpu_cl_get_str_error(err));

    logging_log(LOG_INFO, "Device[%"PRIu64"] Available: %d", idev, device_avaiable);

    cl_ulong memsize;
    if ((err = clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &memsize, NULL)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not get Device[%"PRIu64"] memory size %d: %s", idev, err, gpu_cl_get_str_error(err));

    logging_log(LOG_INFO, "Device[%"PRIu64"] Global Memory: %.4f MB", idev, memsize / (1.0e6));


    cl_ulong mem_allocable;
    if ((err = clGetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &mem_allocable, NULL)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not get Device[%"PRIu64"] allocable memory %d: %s", idev, err, gpu_cl_get_str_error(err));

    logging_log(LOG_INFO, "Device[%"PRIu64"] Allocatable Memory: %.4f MB", idev, mem_allocable / (1.0e6));

    cl_uint maxcomp;
    if ((err = clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &maxcomp, NULL)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not get Device[%"PRIu64"] max compute units %d: %s", idev, err, gpu_cl_get_str_error(err));

    logging_log(LOG_INFO, "Device[%"PRIu64"] Compute Units: %"PRIu64"", idev, maxcomp);

    uint64_t maxworgroup;
    if ((err = clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(uint64_t), &maxworgroup, NULL)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not get Device[%"PRIu64"] work group size %d: %s", idev, err, gpu_cl_get_str_error(err));

    logging_log(LOG_INFO, "Device[%"PRIu64"] Max Work Group Size: %"PRIu64"", idev, maxworgroup);

    cl_uint dimension;
    if ((err = clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &dimension, NULL)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not get Device[%"PRIu64"] max work item dimensions %d: %s", idev, err, gpu_cl_get_str_error(err));

    logging_log(LOG_INFO, "Device[%"PRIu64"] Max Dimension: %"PRIu64"", idev, dimension);

    uint64_t dim_size[dimension];
    if ((err = clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(uint64_t) * dimension, dim_size, NULL)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not get Device[%"PRIu64"] max work items per dimension %d: %s", idev, err, gpu_cl_get_str_error(err));

    char buffer[1024] = {0};
    char *ptr = buffer;
    int adv = 0;

    for (uint64_t i = 0; i < dimension - 1; ++i)
        adv += snprintf(ptr + adv, 1023 - adv, "%"PRIu64", ", dim_size[i]);

    uint64_t i = dimension - 1;
    adv += snprintf(ptr + adv, 1023 - adv, "%"PRIu64"", dim_size[i]);

    logging_log(LOG_INFO, "Device[%"PRIu64"] Max Work Items Per Dimension: {%s}", idev, buffer);
}

static void gpu_cl_init_context(gpu_cl *gpu) {
    cl_int err;
    gpu->ctx = clCreateContext(NULL, gpu->n_devices, gpu->devices, NULL, NULL, &err);
    if (err != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not create context on GPU %d: %s", err, gpu_cl_get_str_error(err));
    logging_log(LOG_INFO, "Created context on GPU");
}

static void gpu_cl_init_queue(gpu_cl *gpu) {
    cl_int err;
#ifdef _WIN32
    gpu->queue = clCreateCommandQueue(gpu->ctx, gpu->devices[d_id], CL_QUEUE_PROFILING_ENABLE, &err);
#else
    cl_command_queue_properties properties[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
        0
    };
    gpu->queue = clCreateCommandQueueWithProperties(gpu->ctx, gpu->devices[d_id], properties, &err);
#endif
    if (err != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not create command queue with profiling properties");
    logging_log(LOG_INFO, "Created command queue on GPU");
}

gpu_cl gpu_cl_init(string current_function, string field_func, string temperature_func, string kernel_augment, string compile_augment) {
    gpu_cl ret = {0};
    gpu_cl_get_platforms(&ret);
    p_id = p_id % ret.n_platforms;

    for (uint64_t i = 0; i < ret.n_platforms; ++i)
        gpu_cl_get_platform_info(ret.platforms[i], i);

    gpu_cl_get_devices(&ret);
    d_id = d_id % ret.n_devices;
    for (uint64_t i = 0; i < ret.n_devices; ++i)
        gpu_cl_get_device_info(ret.devices[i], i);

    gpu_cl_init_context(&ret);
    gpu_cl_init_queue(&ret);

    const char *cmp_ = "-DOPENCL_COMPILATION";
    string cmp = str_is_cstr(cmp_);

    string kernel = fill_functions_on_kernel(current_function, field_func, temperature_func, kernel_augment);
    string compile = fill_compilation_params(cmp, compile_augment);
    gpu_cl_compile_source(&ret, kernel, compile);

    str_free(&kernel);
    str_free(&compile);

    return ret;
}

void gpu_cl_close(gpu_cl *gpu) {
    cl_int err;
    for (uint64_t i = 0; i < gpu->n_kernels; ++i)
        if ((err = clReleaseKernel(gpu->kernels[i].kernel)) != CL_SUCCESS)
            logging_log(LOG_FATAL, "Could not release kernel \"%s\" %d: %s", gpu->kernels[i].name, err, gpu_cl_get_str_error(err));
    mfree(gpu->kernels);

    if ((err = clReleaseProgram(gpu->program)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not release program");

    if ((err = clReleaseCommandQueue(gpu->queue)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not release command queue");

    if ((err = clReleaseContext(gpu->ctx)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not release context");

    for (uint64_t i = 0; i < gpu->n_devices; ++i)
        if ((err = clReleaseDevice(gpu->devices[i])) != CL_SUCCESS)
            logging_log(LOG_FATAL, "Could not release device %zu", i);

    mfree(gpu->devices);
    mfree(gpu->platforms);
    memset(gpu, 0, sizeof(*gpu));
}

uint64_t gpu_cl_append_kernel(gpu_cl *gpu, const char *kernel) {
    cl_int err;
    cl_kernel temp = clCreateKernel(gpu->program, kernel, &err);

    if (err != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not append kernel \"%s\" %d: %s", kernel, err, gpu_cl_get_str_error(err));

    gpu->kernels = mrealloc(gpu->kernels, sizeof(*gpu->kernels) * (gpu->n_kernels + 1));
    gpu->kernels[gpu->n_kernels].name = kernel;
    gpu->kernels[gpu->n_kernels].kernel = temp;
    uint64_t index = gpu->n_kernels;
    ++gpu->n_kernels;
    return index;
}

void gpu_cl_fill_kernel_args(gpu_cl *gpu, uint64_t kernel, uint64_t offset, uint64_t nargs, ...) {
    va_list arg_list;
    va_start(arg_list, nargs);
    cl_int err;
    for (uint64_t i = offset; i < offset + nargs; ++i) {
        void *item = (void*)va_arg(arg_list,  uint64_t);
        uint64_t sz = va_arg(arg_list, uint64_t);
        if ((err = clSetKernelArg(gpu->kernels[kernel].kernel, i, sz, item)) != CL_SUCCESS)
            logging_log(LOG_FATAL, "Could not set argument %d of kernel \"%s\" %d: %s", (int)i, gpu->kernels[kernel].name, err, gpu_cl_get_str_error(err));
    }
    va_end(arg_list);
}

void gpu_cl_enqueue_nd_profiling(gpu_cl *gpu, uint64_t kernel, uint64_t n_dim, uint64_t *local, uint64_t *global, uint64_t *offset) {
    cl_event ev;
    cl_int err;
    if ((err = clEnqueueNDRangeKernel(gpu->queue, gpu->kernels[kernel].kernel, n_dim, offset, global, local, 0, NULL, &ev)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not enqueue kernel \"%s\" %d: %s", gpu->kernels[kernel].name, err, gpu_cl_get_str_error(err));

    uint64_t start, end, duration;
    uint64_t size;
    if ((err = clWaitForEvents(1, &ev)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not wait for kernel \"%s\" event %d: %s", gpu->kernels[kernel].name, err, gpu_cl_get_str_error(err));

    if ((err = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(start), &start, &size)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not get command start for kernel \"%s\" %d: %s", gpu->kernels[kernel].name, err, gpu_cl_get_str_error(err));

    if ((err = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(end), &end, &size)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not get command end for kernel \"%s\" %d: %s", gpu->kernels[kernel].name, err, gpu_cl_get_str_error(err));

    duration = end - start;
    fprintf(stdout, "%s: %e us\n", gpu->kernels[kernel].name, (double)duration / 1000.0);
    if ((err = clReleaseEvent(ev)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not release event for kernel \"%s\" %d: %s", gpu->kernels[kernel].name, err, gpu_cl_get_str_error(err));
}

void gpu_cl_enqueue_nd_no_profiling(gpu_cl *gpu, uint64_t kernel, uint64_t n_dim, uint64_t *local, uint64_t *global, uint64_t *offset) {
    cl_int err;
    if ((err = clEnqueueNDRangeKernel(gpu->queue, gpu->kernels[kernel].kernel, n_dim, offset, global, local, 0, NULL, NULL)) != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not enqueue kernel \"%s\" %d: %s", gpu->kernels[kernel].name, err, gpu_cl_get_str_error(err));
}

cl_mem gpu_cl_create_buffer_base(gpu_cl *gpu, uint64_t size, cl_mem_flags flags, const char *file, int line) {
    cl_int err;
    cl_mem ret = clCreateBuffer(gpu->ctx, flags, size, NULL, &err);
    if (err != CL_SUCCESS)
        logging_log(LOG_FATAL, "On file %s line %d: Could not create buffer with size %"PRIu64" bytes %d: %s", file, line, size, err, gpu_cl_get_str_error(err));

    logging_log(LOG_INFO, "On file %s line %d: Created buffer with size %"PRIu64" bytes", file, line, size);
    return ret;
}

void gpu_cl_write_buffer_base(gpu_cl *gpu, uint64_t size, uint64_t offset, void *host, cl_mem device, const char *name, const char *file, int line) {
    cl_int err = clEnqueueWriteBuffer(gpu->queue, device, CL_TRUE, offset, size, host, 0, NULL, NULL);
    if (err != CL_SUCCESS)
        logging_log(LOG_FATAL, "On file %s line %d: Could not write to GPU buffer \"%s\" %d: %s", file, line, name, err, gpu_cl_get_str_error(err));
}

void gpu_cl_read_buffer_base(gpu_cl *gpu, uint64_t size, uint64_t offset, void *host, cl_mem device, const char *name, const char *file, int line) {
    cl_int err = clEnqueueReadBuffer(gpu->queue, device, CL_TRUE, offset, size, host, 0, NULL, NULL);
    if (err != CL_SUCCESS)
        logging_log(LOG_FATAL, "On file %s line %d: Could not read from GPU buffer \"%s\" %d: %s", file, line, name, err, gpu_cl_get_str_error(err));
}

void gpu_cl_set_kernel_arg(gpu_cl *gpu, uint64_t kernel, uint64_t index, uint64_t size, void *data) {
    cl_int err = clSetKernelArg(gpu->kernels[kernel].kernel, index, size, data);
    if (err != CL_SUCCESS)
        logging_log(LOG_FATAL, "Could not set argument %d of kernel \"%s\" %d: %s", (int)index, gpu->kernels[kernel].name, err, gpu_cl_get_str_error(err));
}

uint64_t gpu_cl_gcd(uint64_t a, uint64_t b) {
    if (b == 0)
        return a;
    else
        return gpu_cl_gcd(b, a % b);
}

void gpu_cl_release_memory_base(cl_mem mem, const char *name, const char *file, int line) {
    cl_int err = clReleaseMemObject(mem);
    if (err != CL_SUCCESS)
        logging_log(LOG_FATAL, "On file %s line %d: could not release memory buffer \"%s\" from GPU %d: %s", file, line, name, err, gpu_cl_get_str_error(err));
    logging_log(LOG_INFO, "On file %s line %d: released memory buffer \"%s\" from GPU", file, line, name);
}
