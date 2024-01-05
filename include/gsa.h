#ifndef __GSA_H
#define __GSA_H
#include "grid_funcs.h"
#include "string_view.h"
#include <stdint.h>

#define gsa(g, ...) gsa_param(g, (gsa_parameters){.qA = 2.8,\
                                                  .qV = 2.6,\
                                                  .qT = 2.2,\
                                                  .T0 = 10.0,\
                                                  .inner_steps = 100000,\
                                                  .outer_steps = 15,\
                                                  .print_factor = 10,\
                                                  .field_function = (string_view){0},\
                                                  .compile_augment = (string_view){0},\
                                                  .kernel_augment = (string_view){0},\
                                                  __VA_ARGS__})

#define gsa_context_init(g, gpu, ...) gsa_context_init_params(g, gpu, (gsa_parameters){.qA = 2.8,\
                                                                                       .qV = 2.6,\
                                                                                       .qT = 2.2,\
                                                                                       .T0 = 10.0,\
                                                                                       .inner_steps = 100000,\
                                                                                       .outer_steps = 15,\
                                                                                       .print_factor = 10,\
                                                                                       .field_function = (string_view){0},\
                                                                                       .compile_augment = (string_view){0},\
                                                                                       .kernel_augment = (string_view){0},\
                                                                                       __VA_ARGS__})

typedef struct {
    double qA;
    double qV;
    double qT;
    double T0;
    uint64_t inner_steps;
    uint64_t outer_steps;
    uint64_t print_factor;

    string_view field_function;
    string_view compile_augment;
    string_view kernel_augment;
} gsa_parameters;

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
    gsa_parameters parameters;

    double qA1;
    double qV1;
    double qT1;
    double oneqA1;
    double exp1;
    double exp2;
    double Tqt;
} gsa_context;

gsa_context gsa_context_init_params(grid *g, gpu_cl *gpu, gsa_parameters param);
gsa_context gsa_context_init_base(grid *g, gpu_cl *gpu, double qA, double qV, double qT, double T0, uint64_t inner_steps, uint64_t outer_steps, uint64_t print_param, string_view field_function, string_view compile_augment, string_view kernel_augment);
void gsa_context_clear(gsa_context *ctx);

void gsa_params(grid *g, gsa_parameters param);
void gsa_base(grid *g, double qA, double qV, double qT, double T0, uint64_t inner_steps, uint64_t outer_steps, uint64_t print_param, string_view field_function, string_view compile_augment, string_view kernel_augment);

void gsa_thermal_step(gsa_context *ctx);
void gsa_metropolis_step(gsa_context *ctx);

#endif
