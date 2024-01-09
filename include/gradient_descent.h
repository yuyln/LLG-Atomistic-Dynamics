#ifndef __GRADIENT_DESCENT_H
#define __GRADIENT_DESCENT_H
#include "gpu.h"
#include "grid_types.h"
#include "grid_funcs.h"

#define gradient_descent_context_init(g, gpu, ...)  gradient_descent_context_init_params(g, gpu, (gradient_descent_param){.T = 0.0,\
                                                                                                           .mass = 1.0,\
                                                                                                           .dt = 1.0e-15,\
                                                                                                           .damping = 1.0,\
                                                                                                           .restoring = 10.0,\
                                                                                                           .T_factor = 0.99,\
                                                                                                           __VA_ARGS__})


typedef struct {
    double T;
    double mass;
    double dt;
    double damping;
    double restoring;
    double T_factor;
} gradient_descent_param;

typedef struct {
    grid *g;
    gpu_cl *gpu;

    cl_mem before_gpu;
    cl_mem after_gpu;
    cl_mem min_gpu;

    double *energy_cpu;
    double min_energy;
    cl_mem energy_gpu;

    double T;
    double mass;
    double dt;
    double damping;
    double restoring;
    double T_factor;

    uint64_t step_id;
    uint64_t exchange_id;
    uint64_t energy_id;

    uint64_t global;
    uint64_t local;
} gradient_descent_context;

gradient_descent_context gradient_descent_context_init_params(grid *g, gpu_cl *gpu, gradient_descent_param param);
gradient_descent_context gradient_descent_context_init_base(grid *g, gpu_cl *gpu, double T, double mass, double dt, double damping, double restoring, double T_factor);

void gradient_descent_step(gradient_descent_context *ctx);
void gradient_descent_clear(gradient_descent_context *ctx);
void gradient_descent_exchange(gradient_descent_context *ctx);
void gradient_descente_read_mininum_grid(gradient_descent_context *ctx);
#endif
