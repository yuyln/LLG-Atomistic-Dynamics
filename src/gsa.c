#include "gsa.h"
#include "constants.h"
#include "grid_funcs.h"
#include "kernel_funcs.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>

static double energy_from_gsa_context(gsa_context *ctx) {
    cl_event ev = clw_enqueue_nd(ctx->gpu->queue, ctx->gpu->kernels[ctx->energy_id], 1, NULL, &ctx->global, &ctx->local);
    clw_print_cl_error(stderr, clEnqueueReadBuffer(ctx->gpu->queue, ctx->energy_gpu, CL_TRUE, 0, ctx->g->gi.rows * ctx->g->gi.cols * sizeof(*ctx->energy_cpu), ctx->energy_cpu, 0, NULL, NULL), "[ FATAL ] Could not read energy buffer GSA");
    double ret = 0.0;
    for (uint64_t i = 0; i < ctx->g->gi.rows * ctx->g->gi.cols; ++i)
        ret += ctx->energy_cpu[i];
    gpu_profiling(stdout, ev, "Calculate Energy");
    return ret;
}

gsa_context gsa_context_init_params(grid *g, gpu_cl *gpu, gsa_parameters param) {
    return gsa_context_init_base(g, gpu, param.qA, param.qV, param.qT, param.T0, param.inner_steps, param.outer_steps, param.print_factor, param.field_function, param.compile_augment, param.kernel_augment);
}

gsa_context gsa_context_init_base(grid *g, gpu_cl *gpu, double qA, double qV, double qT, double T0, uint64_t inner_steps, uint64_t outer_steps, uint64_t print_factor, string_view field_function, string_view compile_augment, string_view kernel_augment) {
    gsa_context ret = {0};
    ret.g = g;
    ret.gpu = gpu;
    ret.energy_cpu = calloc(g->gi.rows * g->gi.cols, sizeof(*ret.energy_cpu));
    ret.outer_step = 0;
    ret.inner_step = 0;
    ret.step = 0;
    ret.T = T0;
    ret.parameters.qA = qA;
    ret.parameters.qV = qV;
    ret.parameters.qT = qT;
    ret.parameters.T0 = T0;
    ret.parameters.inner_steps = inner_steps;
    ret.parameters.outer_steps = outer_steps;
    ret.parameters.print_factor = print_factor;
    ret.parameters.field_function = field_function;
    ret.parameters.compile_augment = compile_augment;
    ret.parameters.kernel_augment = kernel_augment;
    ret.global = g->gi.rows * g->gi.cols;
    ret.local = clw_gcd(ret.global, 32);

    cl_int err;
    ret.energy_gpu = clCreateBuffer(gpu->ctx, CL_MEM_READ_WRITE, g->gi.rows * g->gi.cols * sizeof(*ret.energy_cpu), NULL, &err);
    clw_print_cl_error(stderr, err, "[ FATAL ] Could not create energy buffer");

    ret.swap_gpu = clCreateBuffer(gpu->ctx, CL_MEM_READ_WRITE, g->gi.rows * g->gi.cols * sizeof(*g->m), NULL, &err);
    clw_print_cl_error(stderr, err, "[ FATAL ] Could not create swap buffer gsa");

    ret.min_gpu = clCreateBuffer(gpu->ctx, CL_MEM_READ_WRITE, g->gi.rows * g->gi.cols * sizeof(*g->m), NULL, &err);
    clw_print_cl_error(stderr, err, "[ FATAL ] Could not create min buffer gsa");

    ret.thermal_id = gpu_append_kernel(gpu, "thermal_step_gsa");
    ret.exchange_id = gpu_append_kernel(gpu, "exchange_grid");
    ret.energy_id = gpu_append_kernel(gpu, "calculate_energy");

    ret.qA1 = qA - 1.0;
    ret.qV1 = qV - 1.0;
    ret.qT1 = qT - 1.0;
    ret.oneqA1 = 1.0 / ret.qA1;
    ret.exp1 = 2.0 / (3.0 - qV);
    ret.exp2 = 1.0 / ret.qV1 - 0.5;
    ret.Tqt = T0 * (pow(2.0, ret.qT1) - 1.0);
    ret.gamma = gamma(1.0 / (qV - 1.0)) / gamma(1.0 / (qV - 1.0) - 1.0 / 2.0);

    gpu_fill_kernel_args(gpu, ret.thermal_id, 0, 7, &g->gp_buffer, sizeof(cl_mem), &g->m_buffer, sizeof(cl_mem), &ret.swap_gpu, sizeof(cl_mem), &g->gi, sizeof(grid_info), &ret.parameters.qV, sizeof(double), &ret.gamma, sizeof(double), &ret.T, sizeof(double));

    double t = 0.0;
    gpu_fill_kernel_args(gpu, ret.energy_id, 0, 5, &g->gp_buffer, sizeof(cl_mem), &ret.swap_gpu, sizeof(cl_mem), &g->gi, sizeof(grid_info), &ret.energy_gpu, sizeof(cl_mem), &t, sizeof(double));

    clw_print_cl_error(stderr, clSetKernelArg(gpu->kernels[ret.exchange_id].kernel, 1, sizeof(cl_mem), &ret.swap_gpu), "[ FATAL ] Could not set argument 1 of kernel %s", gpu->kernels[ret.exchange_id].name);


    clw_print_cl_error(stderr, clSetKernelArg(ret.gpu->kernels[ret.exchange_id].kernel, 1, sizeof(cl_mem), &ret.g->m_buffer), "[ FATAL ] HAHAHAHHAHAHAHAHAHAHA WTF0");
    clw_print_cl_error(stderr, clSetKernelArg(ret.gpu->kernels[ret.exchange_id].kernel, 0, sizeof(cl_mem), &ret.swap_gpu), "[ FATAL ] HAHAHAHAHA WTF");
    clw_enqueue_nd(ret.gpu->queue, ret.gpu->kernels[ret.exchange_id], 1, NULL, &ret.global, &ret.local);

    clw_print_cl_error(stderr, clSetKernelArg(gpu->kernels[ret.exchange_id].kernel, 1, sizeof(cl_mem), &ret.swap_gpu), "[ FATAL ] Could not set argument 1 of kernel %s", gpu->kernels[ret.exchange_id].name);


    ret.last_energy = energy_from_gsa_context(&ret);
    ret.min_energy = energy_from_gsa_context(&ret);
    return ret;
}

void gsa_params(grid *g, gsa_parameters param) {
    gsa_base(g, param.qA, param.qV, param.qT, param.T0, param.inner_steps, param.outer_steps, param.print_factor, param.field_function, param.compile_augment, param.kernel_augment);
}

void gsa_base(grid *g, double qA, double qV, double qT, double T0, uint64_t inner_steps, uint64_t outer_steps, uint64_t print_param, string_view field_function, string_view compile_augment, string_view kernel_augment) {
    gpu_cl gpu = gpu_cl_init(0, 0);
    grid_to_gpu(g, gpu);

    const char cmp[] = "-DOPENCL_COMPILATION";
    string kernel = fill_functions_on_kernel(sv_from_cstr("return (current){0};"), field_function, kernel_augment);
    string compile = fill_compilation_params(sv_from_cstr(cmp), compile_augment);
    gpu_cl_compile_source(&gpu, sv_from_cstr(string_as_cstr(&kernel)), sv_from_cstr(string_as_cstr(&compile)));
    string_free(&kernel);
    string_free(&compile);

    gsa_context ctx = gsa_context_init_base(g, &gpu, qA, qV, qT, T0, inner_steps, outer_steps, print_param, field_function, compile_augment, kernel_augment);
    while (ctx.outer_step < outer_steps) {
        while (ctx.inner_step < inner_steps) {
            gsa_thermal_step(&ctx);
            gsa_metropolis_step(&ctx);
        }
    }

    clw_print_cl_error(stderr, clSetKernelArg(ctx.gpu->kernels[ctx.exchange_id].kernel, 0, sizeof(cl_mem), &ctx.g->m_buffer), "[ FATAL ] Could not set min grid as argument of exchange grids GSA");
    clw_print_cl_error(stderr, clSetKernelArg(ctx.gpu->kernels[ctx.exchange_id].kernel, 1, sizeof(cl_mem), &ctx.min_gpu), "[ FATAL ] Could not set min grid as argument of exchange grids GSA");
    clw_enqueue_nd(ctx.gpu->queue, ctx.gpu->kernels[ctx.exchange_id], 1, NULL, &ctx.global, &ctx.local);

    gsa_context_read_minimun_grid(&ctx);

    printf("GSA Done. Minimun energy obtained %.15e eV\n", ctx.min_energy / QE);
}

void gsa_thermal_step(gsa_context *ctx) {
    clw_set_kernel_arg(ctx->gpu->kernels[ctx->thermal_id], 6, sizeof(double), &ctx->T);
    int seed = rand();
    clw_set_kernel_arg(ctx->gpu->kernels[ctx->thermal_id], 7, sizeof(int), &seed);
    cl_event ev = clw_enqueue_nd(ctx->gpu->queue, ctx->gpu->kernels[ctx->thermal_id], 1, NULL, &ctx->global, &ctx->local);
    ctx->step++;
    ctx->inner_step++;
    ctx->T = ctx->Tqt / (pow(ctx->inner_step + 1.0, ctx->qT1) - 1.0);

    if (ctx->step % ctx->parameters.inner_steps == 0) {
        srand(time(NULL));
        ctx->outer_step++;
        ctx->inner_step = 0;
    }

    gpu_profiling(stdout, ev, "Thermal Step");
}

void gsa_metropolis_step(gsa_context *ctx) {
    double new_energy = energy_from_gsa_context(ctx);

    if (new_energy <= ctx->min_energy) {
        ctx->min_energy = new_energy;
        clw_print_cl_error(stderr, clSetKernelArg(ctx->gpu->kernels[ctx->exchange_id].kernel, 0, sizeof(cl_mem), &ctx->min_gpu), "[ FATAL ] Could not set min grid as argument of exchange grids GSA");
        cl_event ev = clw_enqueue_nd(ctx->gpu->queue, ctx->gpu->kernels[ctx->exchange_id], 1, NULL, &ctx->global, &ctx->local);
        gpu_profiling(stdout, ev, "Exchange min grid");

    }

    if (new_energy <= ctx->last_energy) {
        ctx->last_energy = new_energy;
        clw_print_cl_error(stderr, clSetKernelArg(ctx->gpu->kernels[ctx->exchange_id].kernel, 0, sizeof(cl_mem), &ctx->g->m_buffer), "[ FATAL ] Could not set old grid as argument of exchange grids GSA");
        cl_event ev = clw_enqueue_nd(ctx->gpu->queue, ctx->gpu->kernels[ctx->exchange_id], 1, NULL, &ctx->global, &ctx->local);
        gpu_profiling(stdout, ev, "Exchange old grid");
    } else {
        double df = (new_energy - ctx->last_energy) / KB;// / fabs(ctx->g->gp->exchange);
        double pqa = 1.0 / pow(1.0 + ctx->qA1 * df / (ctx->T), ctx->oneqA1);
        if (shit_random(0.0, 1.0) < pqa) {
            ctx->last_energy = new_energy;
            clw_print_cl_error(stderr, clSetKernelArg(ctx->gpu->kernels[ctx->exchange_id].kernel, 0, sizeof(cl_mem), &ctx->g->m_buffer), "[ FATAL ] Could not set old grid as argument of exchange grids GSA");
            cl_event ev = clw_enqueue_nd(ctx->gpu->queue, ctx->gpu->kernels[ctx->exchange_id], 1, NULL, &ctx->global, &ctx->local);
            gpu_profiling(stdout, ev, "Exchange old pqa grid");
        }
    }

    if (ctx->inner_step % (ctx->parameters.inner_steps / ctx->parameters.print_factor) == 0)
        printf("------------------------------------\n"\
               "GSA Outer Step: %zu\n"\
               "GSA Inner Step: %zu\n"\
               "Minimun Energy = %.15e eV\n"\
               "Temperature = %.15e\n"\
               "------------------------------------\n", ctx->outer_step, ctx->inner_step, ctx->min_energy / QE, ctx->T);
}

void gsa_context_clear(gsa_context *ctx) {
    free(ctx->energy_cpu);
    clw_print_cl_error(stderr, clReleaseMemObject(ctx->swap_gpu), "[ FATAL ] Could not release swap buffer from GPU gsa");
    clw_print_cl_error(stderr, clReleaseMemObject(ctx->min_gpu), "[ FATAL ] Could not release min buffer from GPU gsa");
    clw_print_cl_error(stderr, clReleaseMemObject(ctx->energy_gpu), "[ FATAL ] Could not release energy buffer from GPU gsa");

    memset(ctx, 0, sizeof(*ctx));
}

void gsa_context_read_minimun_grid(gsa_context *ctx) {
    clw_print_cl_error(stderr, clEnqueueReadBuffer(ctx->gpu->queue, ctx->g->m_buffer, CL_TRUE, 0, ctx->g->gi.rows * ctx->g->gi.cols * sizeof(*ctx->g->m), ctx->g->m, 0, NULL, NULL), "[ FATAL ] Could not read minimum buffer from GPU gsa");
}
