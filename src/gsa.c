#include "gsa.h"

gsa_context gsa_context_init_params(grid *g, gpu_cl *gpu, gsa_parameters param) {
    return gsa_context_init_base(g, gpu, param.qA, param.qV, param.qT, param.T0, param.inner_steps, param.outer_steps, param.print_factor, param.field_function, param.compile_augment, param.kernel_augment);
}

gsa_context gsa_context_init_base(grid *g, gpu_cl *gpu, double qA, double qV, double qT, double T0, uint64_t inner_steps, uint64_t outer_steps, uint64_t print_param, string_view field_function, string_view compile_augment, string_view kernel_augment) {

}

void gsa_params(grid *g, gsa_parameters param) {
    return gsa_base(g, param.qA, param.qV, param.qT, param.T0, param.inner_steps, param.outer_steps, param.print_factor, param.field_function, param.compile_augment, param.kernel_augment);
}

void gsa_base(grid *g, double qA, double qV, double qT, double T0, uint64_t inner_steps, uint64_t outer_steps, uint64_t print_param, string_view field_function, string_view compile_augment, string_view kernel_augment) {
}

void gsa_thermal_step(gsa_context *ctx) {
}

void gsa_metropolis_step(gsa_context *ctx) {
}
