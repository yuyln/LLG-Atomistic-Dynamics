#ifndef __GSA_H
#define __GSA_H
#include <stdint.h>

#include "grid_funcs.h"
#include "string_view.h"

typedef struct {
    double qA;
    double qV;
    double qT;
    double T0;
    uint64_t inner_steps;
    uint64_t outer_steps;
    uint64_t print_factor;

    string field_func;
    string compile_augment;
} gsa_params;

typedef struct {
    grid *g;
    gpu_cl *gpu;

    cl_mem swap_gpu;
    cl_mem min_gpu;

    cl_mem energy_gpu;
    double *energy_cpu;

    double last_energy;
    double min_energy;

    uint64_t thermal_id;
    uint64_t exchange_id;
    uint64_t energy_id;

    uint64_t outer_step;
    uint64_t inner_step;
    uint64_t step;
    uint64_t global;
    uint64_t local;

    double T;
    gsa_params parameters;

    double qA1;
    double qV1;
    double qT1;
    double oneqA1;
    double exp1;
    double exp2;
    double Tqt;
    double gamma;
} gsa_context;

gsa_context gsa_context_init(grid *g, gpu_cl *gpu, gsa_params params);
void gsa_context_close(gsa_context *ctx);
void gsa_context_read_minimun_grid(gsa_context *ctx);

gsa_params gsa_params_init(void);

void gsa(grid *g, gsa_params params);

void gsa_thermal_step(gsa_context *ctx);
void gsa_metropolis_step(gsa_context *ctx);

#endif
