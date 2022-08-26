#ifndef __SIM
#define __SIM

#include <helpers.h>

#define kernels_n sizeof(kernels) / sizeof(char*)
static const char* kernels[] = {"TermalStep", "HamiltonianGPU", "Reset"};

typedef struct GPU
{
    cl_platform_id *plats; size_t n_plats; int i_plat;
    cl_device_id *devs; size_t n_devs; int i_dev;
    cl_context ctx;
    cl_command_queue queue;
    cl_program program;
    Kernel *kernels; size_t n_kernels;
} GPU;

void FreeGPU(GPU *g)
{
    for (size_t i = 0; i < g->n_kernels; ++i)
    {
        PrintCLError(stderr, clReleaseKernel(g->kernels[i].kernel), "Error releasing kernel %s", g->kernels[i].name);
    }
    
    if (g->kernels)
        free(g->kernels);

    PrintCLError(stderr, clReleaseProgram(g->program), "Error releasing program");
    PrintCLError(stderr, clReleaseContext(g->ctx), "Error releasing context");
    PrintCLError(stderr, clReleaseCommandQueue(g->queue), "Error releasing queue");
    
    for (size_t i = 0; i < g->n_devs; ++i)
    {
        PrintCLError(stderr, clReleaseDevice(g->devs[i]), "Error releasing device[%zu]", i);
    }
    if (g->devs)
        free(g->devs);
    if (g->plats)
        free(g->plats);
   

}

typedef struct Simulator
{
    size_t n_steps;
    double dt;
    size_t n_cpu;
    size_t write_cut;
    bool write_to_file;
    bool use_gpu;
    GPU gpu;
    Grid g_old;
    Grid g_new;
    cl_mem g_old_buffer, g_new_buffer;
    Vec* grid_out_file;
} Simulator;

Simulator InitSimulator(const char* path)
{
    Simulator ret = {0};
    GridParam param_tmp = {0};
    GetGridParam(path, &param_tmp);

    StartParse(path);

    ret.dt = GetValueDouble("DT");
    ret.n_steps = (size_t)GetValueULLInt("STEPS", 10);

    ret.write_to_file = (bool)GetValueInt("WRITE", 10);
    ret.write_cut = (size_t)GetValueULLInt("CUT", 10);
    ret.grid_out_file = (Vec*)calloc(ret.write_to_file * ret.n_steps / ret.write_cut, sizeof(Vec));

    ret.n_cpu = (size_t)GetValueULLInt("CPU", 10);

    bool start_random = (bool)GetValueInt("RSG", 10);
    char* local_file_dir = strdup(parser_global_state[FindIndexOfTag("FILE") + 1]);
    // EndParse();

    if (start_random)
        ret.g_old = InitGridRandom(GetValueInt("ROWS", 10), GetValueInt("COLS", 10));
    else
        ret.g_old = InitGridFromFile(local_file_dir);
    
    free(local_file_dir);

    memcpy(&ret.g_old.param.exchange, &param_tmp.exchange, sizeof(GridParam) - (sizeof(int) * 2 + sizeof(size_t)));

    /*
        Add start to anisotropy and pinning
    */
    CopyGrid(&ret.g_new, &ret.g_old);

    StartParse(path);
    ret.use_gpu = (bool)GetValueInt("GPU", 10);
    if (ret.use_gpu)
    {
        ret.gpu.i_plat = GetValueInt("PLAT", 10);
        ret.gpu.i_dev = GetValueInt("DEV", 10);
        ret.gpu.plats = InitPlatforms(&ret.gpu.n_plats);

        for (size_t i = 0; i < ret.gpu.n_plats; ++i)
            PlatformInfo(stdout, ret.gpu.plats[i], i);

        ret.gpu.devs = InitDevices(ret.gpu.plats[ret.gpu.i_plat], &ret.gpu.n_devs);
        for (size_t i = 0; i < ret.gpu.n_devs; ++i)
            DeviceInfo(stdout, ret.gpu.devs[i], i);


        ret.gpu.ctx = InitContext(ret.gpu.devs, ret.gpu.n_devs);
        ret.gpu.queue = InitQueue(ret.gpu.ctx, ret.gpu.devs[ret.gpu.i_dev]);
        ret.gpu.program = InitProgramSource(ret.gpu.ctx, kernel_data);

        char* comp_opt;
        size_t comp_opt_size = snprintf(NULL, 0, "-I ./headers -DROWS=%d -DCOLS=%d -DTOTAL=%zu -DOPENCLCOMP", ret.g_old.param.rows, ret.g_old.param.cols, ret.g_old.param.total) + 1;
        comp_opt = (char*)calloc(comp_opt_size, 1);
        snprintf(comp_opt, comp_opt_size, "-I ./headers -DROWS=%d -DCOLS=%d -DTOTAL=%zu -DOPENCLCOMP", ret.g_old.param.rows, ret.g_old.param.cols, ret.g_old.param.total);
        printf("Compile OpenCL: %s\n", comp_opt);

        cl_int err = BuildProgram(ret.gpu.program, ret.gpu.n_devs, ret.gpu.devs, comp_opt);
        BuildProgramInfo(stdout, ret.gpu.program, ret.gpu.devs[ret.gpu.i_dev], err);

        free(comp_opt);
        ret.g_old_buffer = CreateBuffer(FindGridSize(&ret.g_old), ret.gpu.ctx, CL_MEM_READ_WRITE);
        ret.g_new_buffer = CreateBuffer(FindGridSize(&ret.g_new), ret.gpu.ctx, CL_MEM_READ_WRITE);
        WriteFullGridBuffer(ret.gpu.queue, ret.g_old_buffer, &ret.g_old);
        WriteFullGridBuffer(ret.gpu.queue, ret.g_new_buffer, &ret.g_new);
        ret.gpu.kernels = InitKernels(ret.gpu.program, kernels, kernels_n);
    }

    EndParse();
    return ret;
}

void FreeSimulator(Simulator *s)
{
    if (s->grid_out_file)
        free(s->grid_out_file);
    
    FreeGrid(&s->g_old);
    FreeGrid(&s->g_new);
    if (s->use_gpu)
    {
        PrintCLError(stderr, clReleaseMemObject(s->g_old_buffer), "Error releasing g_old buffer");
        PrintCLError(stderr, clReleaseMemObject(s->g_new_buffer), "Error releasing g_new buffer");
        FreeGPU(&s->gpu);
    }

}

void ExportSimulator(Simulator* s, FILE* file)
{
    fprintf(file, "Times Steps: %zu\n", s->n_steps);
    fprintf(file, "Time Step: %e\n", s->dt);
    fprintf(file, "Exchange: %e\n", s->g_old.param.exchange);
    fprintf(file, "DMI: %e\n", s->g_old.param.dm);
}

void ExportSimulatorFile(Simulator* s, const char* path)
{
    FILE *file = fopen(path, "w");
    if (!file)
    {
        fprintf(stderr, "Could not open file %s: %s\n", path, strerror(errno));
        exit(1);
    }
    fprintf(file, "Times Steps: %zu\n", s->n_steps);
    fprintf(file, "Time Step: %e\n", s->dt);
    fprintf(file, "Exchange: %e\n", s->g_old.param.exchange);
    fprintf(file, "DMI: %e\n", s->g_old.param.dm);
    fclose(file);
}
#endif