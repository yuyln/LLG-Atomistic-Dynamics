#ifndef __INTEGRATE_H
#define __INTEGRATE_H
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#include "gpu.h"
#include "grid_funcs.h"
#include "constants.h"
#include "complete_kernel.h"
#include "colors.h"

typedef struct {
    double dt;
    double duration;

    unsigned int interval_for_information;
    unsigned int interval_for_raw_grid;
    unsigned int interval_for_rgb_grid;
    unsigned int interval_for_cluster;

    const char *current_func;
    const char *field_func;
    const char *temperature_func;
    const char *compile_augment;
    const char *output_path;

    bool do_cluster;
    double(*cluster_metric)(grid*, uint64_t, uint64_t, uint64_t, uint64_t, void*);
    double(*cluster_weight)(grid*, uint64_t, uint64_t, void*);
    void *cluster_metric_data;
    void *cluster_weight_data;

    double cluster_eps;
    uint64_t cluster_min_pts;
} integrate_params;

typedef struct {
    grid *g;
    gpu_cl *gpu;

    integrate_params params;
    double time;
    uint64_t integrate_step;

    cl_mem swap_gpu;
    uint64_t step_id;
    uint64_t exchange_id;
    uint64_t global;
    uint64_t local;

    FILE *integrate_info;
    FILE *integrate_evolution;
    FILE *clusters;

    information_packed *info;
    cl_mem info_gpu;
    uint64_t info_id;

    RGBA32 *rgb;
    cl_mem rgb_gpu;
    uint64_t render_id;
} integrate_context;

integrate_context integrate_context_init(grid *grid, gpu_cl *gpu, integrate_params dt);
void integrate_context_close(integrate_context *ctx);
void integrate_context_read_grid(integrate_context *ctx);

integrate_params integrate_params_init(void);
void integrate(grid *g, integrate_params params);

void integrate_step(integrate_context *ctx);
void integrate_exchange_grids(integrate_context *ctx);
information_packed integrate_get_info(integrate_context *ctx);

#endif
