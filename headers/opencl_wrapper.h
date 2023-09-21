#ifndef __OPENCLW
#define __OPENCLW
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

//https://github.com/yuyln/OpenCLWrapper/blob/master/OpenCLWrapper.h

/*
    #if defined(unix) || defined(__unix) || defined(__unix)
    setenv("CUDA_CACHE_DISABLE", "1", 1);
    #elif defined(_WIN32) || defined(_WIN64) || defined(__CYGWIN__)
    _putenv_s("CUDA_CACHE_DISABLE", "1");
    #elif defined(__APPLE__) || defined(__MACH__)
    setenv("CUDA_CACHE_DISABLE", "1", 1);
    #endif
*/

static const char *errors[60] = {
                                "CL_SUCCESS",
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
                                "CL_INVALID_DEVICE_PARTITION_COUNT",
                            };

#define PrintCLError(file, err, message, ...) {                                                                                             \
                                            if (err != 0)                                                                                        \
                                            {                                                                                                    \
                                                fprintf((file), "%s: %d OPENCLWRAPPER ERROR %d: ", __FILE__, __LINE__, err);\
                                                int err_ = abs((err));\
                                                if (err_ >= 30)\
                                                    err_ = err_ - 10;\
                                                fprintf((file), "%s | ", errors[err_]);\
                                                fprintf((file), (message), ##__VA_ARGS__);                                                                  \
                                                fprintf((file), "\n");                                                                             \
                                                exit((err));                                                                                       \
                                            }                                                                                                    \
                                          }

    typedef struct
    {
        cl_kernel kernel;
        const char *name;
    } Kernel;

    //utils
    int ReadFileW(const char *path, char **out);
    size_t gcd(size_t a, size_t b);
    // long unsigned int gcd(long unsigned int a, long unsigned int b);

    //buffer related
    cl_mem CreateBuffer(size_t size, cl_context context, cl_mem_flags flags);
    void ReadBuffer(cl_mem buffer, void *host_ptr, size_t size, size_t offset, cl_command_queue q);
    void WriteBuffer(cl_mem buffer, void *host_ptr, size_t size, size_t offset, cl_command_queue q);

    //init related
    cl_platform_id* InitPlatforms(size_t *n);
    cl_device_id* InitDevices(cl_platform_id plat, size_t *n);
    cl_context InitContext(cl_device_id *devices, size_t ndev);
    cl_command_queue InitQueue(cl_context ctx, cl_device_id device);
    cl_program InitProgramSource(cl_context ctx, const char *source);
    cl_int BuildProgram(cl_program program, size_t ndev, cl_device_id *devs, const char *compile_opt);
    Kernel InitKernel(cl_program program, const char *name);
    Kernel *InitKernels(cl_program program, const char **names, size_t n);

    //info related
    void PlatformInfo(FILE *file, cl_platform_id plat, size_t iplat);
    void DeviceInfo(FILE *file, cl_device_id dev, size_t idev);
    void BuildProgramInfo(FILE *f, cl_program program, cl_device_id dev, cl_int errCode); //errCode from BuildProgram returned call

    //queue related
    void EnqueueND(cl_command_queue queue, Kernel k, size_t dim, size_t *global_offset, size_t *global, size_t *local);
    void Finish(cl_command_queue queue);

    //work related
    size_t LocalWorkDeviceGDC_1D(size_t global, cl_device_id dev);
    size_t LocalWorkGDC_1D(size_t global, size_t fac);
    size_t *LocalWorkDeviceGDC_ND(size_t ndim, size_t *global, cl_device_id dev);
    size_t *LocalWorkGDC_ND(size_t ndim, size_t *global, size_t fac);

    //kernel arg related
    void SetKernelArg(Kernel k, size_t argIndex, size_t argSize, void *arg);

#ifdef OPENCLWRAPPER_IMPLEMENTATION

void PrintCLError_(FILE *f, int err, const char *m, int line, const char *file)
{
    if (err != 0)
    {
        fprintf(f, "%s:%d OPENCLWRAPPER ERROR %s:%d\n", file, line, m, err);
        exit(err);
    }
}

int ReadFileW(const char *path, char **out)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL)
    {
        fprintf(stderr, "FILE (%s) IS NULL: %s", path, strerror(errno));
        return -1;
    }
    fseek(f, 0, SEEK_SET);
    fseek(f, 0, SEEK_END);
    size_t filesize = ftell(f);
    fseek(f, 0, SEEK_SET);

    *out = (char*)malloc(filesize + 1);
    size_t readsize = fread((void *)(*out), 1, filesize, f);
    if (readsize != filesize)
    {
        PrintCLError(stderr, -1, "READSIZE AND FILESIZE ARE NOT EQUAL");
    }
    (*out)[filesize] = '\0';
    fclose(f);
    return filesize;
}

size_t gcd(size_t a, size_t b)
{
    if (b == 0)
    {
        return a;
    }
    else
    {
        return gcd(b, a % b);
    }
}

cl_mem CreateBuffer(size_t size, cl_context context, cl_mem_flags flags)
{
    cl_int err;
    cl_mem ret = clCreateBuffer(context, flags, size, NULL, &err);
    PrintCLError(stderr, err, "ERROR ON CREATING BUFFER WITH SIZE %zu", size);
    return ret;
}

void ReadBuffer(cl_mem buffer, void *host_ptr, size_t size, size_t offset, cl_command_queue q)
{
    cl_int err = clEnqueueReadBuffer(q, buffer, CL_TRUE, offset, size, host_ptr, 0, NULL, NULL);
    PrintCLError(stderr, err, "ERROR ON READING BUFFER WITH SIZE %zu AND OFFSET %zu", size, offset);
}

void WriteBuffer(cl_mem buffer, void *host_ptr, size_t size, size_t offset, cl_command_queue q)
{
    cl_int err = clEnqueueWriteBuffer(q, buffer, CL_TRUE, offset, size, host_ptr, 0, NULL, NULL);
    PrintCLError(stderr, err, "ERROR ON WRITING BUFFER WITH SIZE %zu AND OFFSET %zu", size, offset);
}

cl_platform_id* InitPlatforms(size_t *n)
{
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
    PrintCLError(stderr, err, "ERROR FINDING NUMBER OF PLATFORMS");
    *n = nn;

    cl_platform_id *local = (cl_platform_id*)malloc(sizeof(cl_platform_id) * nn);
    err = clGetPlatformIDs(nn, local, NULL);
    PrintCLError(stderr, err, "ERROR INIT PLATFORM");
    return local;
}

cl_device_id* InitDevices(cl_platform_id plat, size_t *n)
{
    cl_uint nn;
    cl_int err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, NULL, &nn);
    PrintCLError(stderr, err, "ERROR FINDING NUMBER OF DEVICES");
    *n = nn;

    cl_device_id *local = (cl_device_id*)malloc(sizeof(cl_device_id) * nn);
    err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, nn, local, NULL);
    PrintCLError(stderr, err, "ERROR INIT PLATFORM");
    return local;
}

cl_context InitContext(cl_device_id *devices, size_t ndev)
{
    cl_int err;
    cl_context ret = clCreateContext(NULL, ndev, devices, NULL, NULL, &err);
    PrintCLError(stderr, err, "ERROR CREATING CONTEXT");
    return ret;    
}

cl_command_queue InitQueue(cl_context ctx, cl_device_id device)
{
    cl_int err;
    cl_command_queue ret = clCreateCommandQueue(ctx, device, 0, &err);
    PrintCLError(stderr, err, "ERROR CREATING QUEUE");
    return ret;
}

cl_program InitProgramSource(cl_context ctx, const char *source)
{
    cl_int err;
    cl_program ret = clCreateProgramWithSource(ctx, 1, &source, NULL, &err);
    PrintCLError(stderr, err, "ERROR CREATING PROGRAM WITH SOURCE");
    return ret;
}

cl_int BuildProgram(cl_program program, size_t ndev, cl_device_id *devs, const char *compile_opt)
{
    return clBuildProgram(program, ndev, devs, compile_opt, NULL, NULL);
}

Kernel InitKernel(cl_program program, const char *name)
{
    cl_int err;
    Kernel ret;
    ret.kernel = clCreateKernel(program, name, &err);
    ret.name = name;
    PrintCLError(stderr, err, "ERROR CREATING KERNEL %s", name);
    return ret;
}

Kernel *InitKernels(cl_program program, const char **names, size_t n)
{
    Kernel *ret = (Kernel*)malloc(sizeof(Kernel) * n);
    for (size_t i = 0; i < n; ++i)
    {
        ret[i] = InitKernel(program, names[i]);
    }
    return ret;
}

void EnqueueND(cl_command_queue queue, Kernel k, size_t dim, size_t *global_offset, size_t *global, size_t *local)
{
    cl_int err = clEnqueueNDRangeKernel(queue, k.kernel, dim, global_offset, global, local, 0, NULL, NULL);
    PrintCLError(stderr, err, "ERROR ENQUEUING KERNEL %s", k.name);
}

void Finish(cl_command_queue queue)
{
    cl_int err = clFinish(queue);
    PrintCLError(stderr, err, "ERROR FINISHING QUEUE");
}

size_t LocalWorkDeviceGDC_1D(size_t global, cl_device_id dev)
{
    size_t devG;
    cl_int err = clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &devG, NULL);
    PrintCLError(stderr, err, "ERROR GETTING DEVICE GCD");
    return gcd(global, devG);
}

size_t LocalWorkGDC_1D(size_t global, size_t fac)
{
    return gcd(global, fac);
}

size_t *LocalWorkDeviceGDC_ND(size_t ndim, size_t *global, cl_device_id dev)
{
    size_t *ret = (size_t*)malloc(sizeof(size_t) * ndim);
    for (size_t i = 0; i < ndim; ++i)
    {
        ret[i] = LocalWorkDeviceGDC_1D(global[i], dev);
    }
    return ret;
}

size_t *LocalWorkGDC_ND(size_t ndim, size_t *global, size_t fac)
{
    size_t *ret = (size_t*)malloc(sizeof(size_t) * ndim);
    for (size_t i = 0; i < ndim; ++i)
    {
        ret[i] = LocalWorkGDC_1D(global[i], fac);
    }
    return ret;
}

void SetKernelArg(Kernel k, size_t argIndex, size_t argSize, void *arg)
{
    cl_int err = clSetKernelArg(k.kernel, argIndex, argSize, arg);
    PrintCLError(stderr, err, "ERROR SETTING ARG %zu ON KERNEL %s", argIndex, k.name);
}

void PlatformInfo(FILE *file, cl_platform_id plat, size_t iplat)
{
    size_t n;
    cl_int err = clGetPlatformInfo(plat, CL_PLATFORM_NAME, 0, NULL, &n);
    PrintCLError(stderr, err, "ERROR GETTING SIZE PLATFORM[%zu] NAME INFO", iplat);

    char *info = (char*)malloc(n);
    err = clGetPlatformInfo(plat, CL_PLATFORM_NAME, n, info, NULL);
    PrintCLError(stderr, err, "ERROR GETTING PLATFORM[%zu] NAME INFO", iplat);
    fprintf(file, "---------------------------------\n");
    fprintf(file, "PLATFORM[%zu] NAME: %s\n", iplat, info);
    fprintf(file, "---------------------------------\n");
    free(info);
}

void DeviceInfo(FILE *file, cl_device_id dev, size_t idev)
{
    fprintf(file, "---------------------------------\n");


    size_t n;
    cl_platform_id plt;

    cl_int err = clGetDeviceInfo(dev, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &plt, NULL);
    PrintCLError(stderr, err, "ERROR GETTING PLATFORM FROM DEVICE[%zu]", idev);

    err = clGetPlatformInfo(plt, CL_PLATFORM_NAME, 0, NULL, &n);
    PrintCLError(stderr, err, "ERROR GETTING PLATFORM SIZE FROM DEVICE[%zu]", idev);

    char *info = (char*)malloc(n);
    err = clGetPlatformInfo(plt, CL_PLATFORM_NAME, n, info, NULL);
    PrintCLError(stderr, err, "ERROR GETTING PLATFORM NAME FROM DEVICE[%zu]", idev);

    fprintf(file, "DEVICE[%zu] ON PLATFORM %s\n", idev, info);
    free(info);

    err = clGetDeviceInfo(dev, CL_DEVICE_VENDOR, 0, NULL, &n);
    PrintCLError(stderr, err, "ERROR GETTING SIZE DEVICE[%zu] VENDOR INFO", idev);
    info = (char*)malloc(n);
    err = clGetDeviceInfo(dev, CL_DEVICE_VENDOR, n, info, NULL);
    PrintCLError(stderr, err, "ERROR GETTING DEVICE[%zu] VENDOR INFO", idev);
    fprintf(file, "DEVICE[%zu] VENDOR: %s\n", idev, info);
    free(info);

    err = clGetDeviceInfo(dev, CL_DEVICE_VERSION, 0, NULL, &n);
    PrintCLError(stderr, err, "ERROR GETTING SIZE DEVICE[%zu] VERSION INFO", idev);
    info = (char*)malloc(n);
    err = clGetDeviceInfo(dev, CL_DEVICE_VERSION, n, info, NULL);
    PrintCLError(stderr, err, "ERROR GETTING DEVICE[%zu] VERSION INFO", idev);
    fprintf(file, "DEVICE[%zu] VERSION: %s\n", idev, info);
    free(info);

    err = clGetDeviceInfo(dev, CL_DRIVER_VERSION, 0, NULL, &n);
    PrintCLError(stderr, err, "ERROR GETTING SIZE DEVICE[%zu] DRIVER VERSION INFO", idev);
    info = (char*)malloc(n);
    err = clGetDeviceInfo(dev, CL_DRIVER_VERSION, n, info, NULL);
    PrintCLError(stderr, err, "ERROR GETTING DEVICE[%zu] DRIVER VERSION INFO", idev);
    fprintf(file, "DEVICE[%zu] DRIVER VERSION: %s\n", idev, info);
    free(info);

    err = clGetDeviceInfo(dev, CL_DEVICE_NAME, 0, NULL, &n);
    PrintCLError(stderr, err, "ERROR GETTING SIZE DEVICE[%zu] NAME INFO", idev);
    info = (char*)malloc(n);
    err = clGetDeviceInfo(dev, CL_DEVICE_NAME, n, info, NULL);
    PrintCLError(stderr, err, "ERROR GETTING DEVICE[%zu] NAME INFO", idev);
    fprintf(file, "DEVICE[%zu] NAME: %s\n", idev, info);
    free(info);

    cl_bool device_avaiable;
    err = clGetDeviceInfo(dev, CL_DEVICE_AVAILABLE, sizeof(cl_bool), &device_avaiable, NULL);
    fprintf(file, "DEVICE[%zu] AVAILABLE: %d\n", idev, device_avaiable);

    cl_ulong memsize;
    err = clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &memsize, NULL);
    PrintCLError(stderr, err, "ERROR GETTING DEVICE[%zu] MEM INFO", idev);
    fprintf(file, "DEVICE[%zu] GLOBAL MEM: %.4f MB\n", idev, memsize / (1e6));


    cl_ulong memalloc;
    err = clGetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &memalloc, NULL);
    PrintCLError(stderr, err, "ERROR GETTING DEVICE[%zu] MEM INFO", idev);
    fprintf(file, "DEVICE[%zu] ALLOCATABLE MEM: %.4f MB\n", idev, memalloc / (1e6));

    cl_uint maxcomp;
    err = clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &maxcomp, NULL);
    PrintCLError(stderr, err, "ERROR GETTING DEVICE[%zu] COMPUTE UNITS INFO", idev);
    fprintf(file, "DEVICE[%zu] COMPUTE UNITS: %u\n", idev, maxcomp);

    size_t maxworgroup;
    err = clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxworgroup, NULL);
    PrintCLError(stderr, err, "ERROR GETTING DEVICE[%zu] WORK GROUP INFO", idev);
    fprintf(file, "DEVICE[%zu] MAX WORK GROUP SIZE: %zu\n", idev, maxworgroup);

    cl_uint dimension;
    err = clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &dimension, NULL);
    PrintCLError(stderr, err, "ERROR GETTING DEVICE[%zu] WORK GROUP INFO", idev);
    fprintf(file, "DEVICE[%zu] MAX DIMENSIONS: %u\n", idev, dimension);

    size_t *dim_size = (size_t*)malloc(sizeof(size_t) * dimension);
    err = clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * dimension, dim_size, NULL);
    PrintCLError(stderr, err, "ERROR GETTING DEVICE[%zu] WORK GROUP PER DIMENSION", idev);

    fprintf(file, "DEVICE[%zu] MAX WORK GROUP SIZE PER DIMENSION: {", idev);

    for (size_t i = 0; i < dimension - 1; ++i)
    {
        fprintf(file, "%zu, ", dim_size[i]);
    }
    size_t i = dimension - 1;
    fprintf(file, "%zu}\n", dim_size[i]);
    free(dim_size);


    fprintf(file, "---------------------------------\n");
}

void BuildProgramInfo(FILE *f, cl_program program, cl_device_id dev, cl_int errCode)
{
    size_t size;
    cl_int err = clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &size);
    PrintCLError(stderr, err, "ERROR GETTING BUILD LOG SIZE");

    char *info = (char*)malloc(size);

    err = clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, size, info, NULL);
    PrintCLError(stderr, err, "ERROR GETTING BUILD LOG");

    fprintf(f, "---------------------------------\n");
    fprintf(f, "BUILD LOG: \n%s\n", info);
    fprintf(f, "---------------------------------\n");
    free(info);
    PrintCLError(stderr, errCode, "ERROR BUILDING PROGRAM");
}

#endif // HEADER_IMPL

#endif //__OPENCLW
