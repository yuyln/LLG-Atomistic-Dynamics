#ifndef __GRADIENT_DESCENT_H
#define __GRADIENT_DESCENT_H
#include "gpu.h"
#include "grid_types.h"
#include "grid_funcs.h"

typedef struct {
    double T;
    double mass;
    double dt;
    double damping;
    double restoring;
    double T_factor;
    string field_func;
    string compile_augment;
} gradient_descent_params;

typedef struct {
    grid *g;
    gpu_cl *gpu;

    cl_mem before_gpu;
    cl_mem after_gpu;
    cl_mem min_gpu;

    double *energy_cpu;
    double min_energy;
    cl_mem energy_gpu;

    gradient_descent_params params;

    uint64_t step_id;
    uint64_t exchange_id;
    uint64_t energy_id;

    uint64_t global;
    uint64_t local;
} gradient_descent_context;

gradient_descent_context gradient_descent_context_init(grid *g, gpu_cl *gpu, gradient_descent_params params);
void gradient_descent(grid *g, gradient_descent_params params);

gradient_descent_params gradient_descent_params_init(void);
void gradient_descent_step(gradient_descent_context *ctx);
void gradient_descent_close(gradient_descent_context *ctx);
void gradient_descent_exchange(gradient_descent_context *ctx);
void gradient_descent_read_mininum_grid(gradient_descent_context *ctx);
#endif
