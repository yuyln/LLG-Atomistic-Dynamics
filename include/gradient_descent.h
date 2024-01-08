#ifndef __GRADIENT_DESCENT_H
#define __GRADIENT_DESCENT_H
#include "gpu.h"
#include "grid_types.h"
#include "grid_funcs.h"

typedef struct {
    grid *g;
    gpu_cl *gpu;

    cl_mem after_gpu;
    cl_mem before_gpu;
    cl_mem min_gpu;

    double *energy_cpu;
    cl_mem energy_gpu;

    double mass;
    double T;
    double time;
    double damping;
    double restoring;
} gradient_descent_context;

void gradient_descent_step(gradient_descent_context *ctx);
void gradient_descente_clear(gradient_descent_context *ctx);
#endif
