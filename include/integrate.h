#ifndef __INTEGRATE_H
#define __INTEGRATE_H
#include <stdint.h>
#include <stdlib.h>

#include "gpu.h"
#include "grid_funcs.h"
#include "string_view.h"
#include "constants.h"
#include "complete_kernel.h"

#define integrate(grid, ...) integrate_vars(grid, (integration_params){\
                                                                       .dt=1.0e-15,\
                                                                       .duration=1.0 * NS,\
                                                                       .interval_for_information=100,\
                                                                       .interval_for_writing_grid=1000,\
                                                                       .current_generation_function=str_is_cstr("return (current){0};"),\
                                                                       .field_generation_function=str_is_cstr("double normalized = 0.5 * gs.dm * gs.dm / gs.exchange;\ndouble real = normalized / gs.mu; return v3d_c(0.0, 0.0, real);"),\
                                                                       .temperature_generation_function=str_is_cstr("return 0.0;"),\
                                                                       .output_path = str_is_cstr("./integration/"),\
                                                                       .compile_augment = (string){0},\
                                                                       __VA_ARGS__})

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

void integrate_vars(grid *g, integration_params param);
void integrate_base(grid *grid, double dt, double duration, unsigned int interval_info, unsigned int interval_grid, string func_current, string func_field, string func_temperature, string dir_out, string compile_augment);

void integrate_step(integrate_context *ctx);
void integrate_exchange_grids(integrate_context *ctx);
void integrate_get_info(integrate_context *ctx, uint64_t info_id);

#endif
