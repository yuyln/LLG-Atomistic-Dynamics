#ifndef __INTEGRATE_H
#define __INTEGRATE_H
#include <stdint.h>
#include <stdlib.h>

#include "gpu.h"
#include "grid_funcs.h"
#include "string_view.h"
#include "constants.h"
#include "complete_kernel.h"

typedef struct {
    double dt;
    double duration;
    unsigned int interval_for_information;
    unsigned int interval_for_writing_grid;
    string current_generation_function;
    string field_generation_function;
    string temperature_generation_function;
    string compile_augment;
    string output_path;
} integration_params;

typedef struct {
    grid *g;
    gpu_cl *gpu;
    double dt;
    double time;
    cl_mem swap_buffer;
    uint64_t step_id;
    uint64_t exchange_id;
    uint64_t global;
    uint64_t local;
} integrate_context;

integrate_context integrate_context_init(grid *grid, gpu_cl *gpu, double dt);
void integrate_context_close(integrate_context *ctx);
void integrate_context_read_grid(integrate_context *ctx);

void integrate(grid *g, integration_params param);

void integrate_step(integrate_context *ctx);
void integrate_exchange_grids(integrate_context *ctx);
void integrate_get_info(integrate_context *ctx, uint64_t info_id);

#endif
