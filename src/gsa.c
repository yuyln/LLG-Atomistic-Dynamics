#include "gsa.h"
#include "constants.h"
#include "grid_funcs.h"
#include "kernel_funcs.h"
#include "logging.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <inttypes.h>

static double energy_from_gsa_context(gsa_context *ctx) {
    gpu_cl_enqueue_nd(ctx->gpu, ctx->energy_id, 1, &ctx->local, &ctx->global, NULL);
    gpu_cl_read_buffer(ctx->gpu, ctx->g->gi.rows * ctx->g->gi.cols * sizeof(*ctx->energy_cpu), 0, ctx->energy_cpu, ctx->energy_gpu);
    double ret = 0.0;
    for (uint64_t i = 0; i < ctx->g->gi.rows * ctx->g->gi.cols; ++i)
        ret += ctx->energy_cpu[i];
    return ret;
}


gsa_context gsa_context_init_base(grid *g, gpu_cl *gpu, double qA, double qV, double qT, double T0, uint64_t inner_steps, uint64_t outer_steps, uint64_t print_factor) {
    gsa_context ret = {0};
    ret.g = g;
    ret.gpu = gpu;

    ret.energy_cpu = calloc(g->gi.rows * g->gi.cols, sizeof(*ret.energy_cpu));
    if (!ret.energy_cpu)
        logging_log(LOG_FATAL, "Could not allocate[%"PRIu64" bytes] memory for energy on cpu %s", g->gi.rows * g->gi.cols * sizeof(*ret.energy_cpu), strerror(errno));

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
    ret.global = g->gi.rows * g->gi.cols;
    ret.local = gpu_cl_gcd(ret.global, 32);
    grid_to_gpu(g, *gpu);

    ret.energy_gpu = gpu_cl_create_buffer(gpu, g->gi.rows * g->gi.cols * sizeof(*ret.energy_cpu), CL_MEM_READ_WRITE);
    ret.swap_gpu = gpu_cl_create_buffer(gpu, g->gi.rows * g->gi.cols * sizeof(*g->m), CL_MEM_READ_WRITE);
    ret.min_gpu = gpu_cl_create_buffer(gpu, g->gi.rows * g->gi.cols * sizeof(*g->m), CL_MEM_READ_WRITE);

    ret.thermal_id = gpu_cl_append_kernel(gpu, "thermal_step_gsa");
    ret.exchange_id = gpu_cl_append_kernel(gpu, "exchange_grid");
    ret.energy_id = gpu_cl_append_kernel(gpu, "calculate_energy");

    ret.qA1 = qA - 1.0;
    ret.qV1 = qV - 1.0;
    ret.qT1 = qT - 1.0;
    ret.oneqA1 = 1.0 / ret.qA1;
    ret.exp1 = 2.0 / (3.0 - qV);
    ret.exp2 = 1.0 / ret.qV1 - 0.5;
    ret.Tqt = T0 * (pow(2.0, ret.qT1) - 1.0);
    ret.gamma = tgamma(1.0 / (qV - 1.0)) / tgamma(1.0 / (qV - 1.0) - 1.0 / 2.0);

    gpu_cl_fill_kernel_args(gpu, ret.thermal_id, 0, 7, &g->gp_buffer, sizeof(cl_mem), &g->m_buffer, sizeof(cl_mem), &ret.swap_gpu, sizeof(cl_mem), &g->gi, sizeof(grid_info), &ret.parameters.qV, sizeof(double), &ret.gamma, sizeof(double), &ret.T, sizeof(double));

    double t = 0.0;
    gpu_cl_fill_kernel_args(gpu, ret.energy_id, 0, 5, &g->gp_buffer, sizeof(cl_mem), &ret.swap_gpu, sizeof(cl_mem), &g->gi, sizeof(grid_info), &ret.energy_gpu, sizeof(cl_mem), &t, sizeof(double));

    gpu_cl_fill_kernel_args(gpu, ret.exchange_id, 0, 2, &ret.swap_gpu, sizeof(cl_mem), &ret.g->m_buffer, sizeof(cl_mem));
    gpu_cl_enqueue_nd(gpu, ret.exchange_id, 1, &ret.local, &ret.global, NULL);

    gpu_cl_fill_kernel_args(gpu, ret.exchange_id, 0, 2, &ret.min_gpu, sizeof(cl_mem), &ret.g->m_buffer, sizeof(cl_mem));
    gpu_cl_enqueue_nd(gpu, ret.exchange_id, 1, &ret.local, &ret.global, NULL);

    gpu_cl_set_kernel_arg(gpu, ret.exchange_id, 1, sizeof(cl_mem), &ret.swap_gpu);

    ret.last_energy = energy_from_gsa_context(&ret);
    ret.min_energy = energy_from_gsa_context(&ret);
    return ret;
}

gsa_context gsa_context_init(grid *g, gpu_cl *gpu, gsa_params params) {
    return gsa_context_init_base(g, gpu, params.qA, params.qV, params.qT, params.T0, params.inner_steps, params.outer_steps, params.print_factor);
}


void gsa_base(grid *g, double qA, double qV, double qT, double T0, uint64_t inner_steps, uint64_t outer_steps, uint64_t print_params, string field_func, string compile_augment) {
    gpu_cl gpu = gpu_cl_init(STR_NULL, field_func, STR_NULL, STR_NULL, compile_augment);
    grid_to_gpu(g, gpu);

    gsa_context ctx = gsa_context_init_base(g, &gpu, qA, qV, qT, T0, inner_steps, outer_steps, print_params);
    while (ctx.outer_step < outer_steps) {
        while (ctx.inner_step < inner_steps) {
            gsa_thermal_step(&ctx);
            gsa_metropolis_step(&ctx);
        }
    }

    gpu_cl_set_kernel_arg(&gpu, ctx.exchange_id, 0, sizeof(cl_mem), &ctx.g->m_buffer);
    gpu_cl_set_kernel_arg(&gpu, ctx.exchange_id, 1, sizeof(cl_mem), &ctx.min_gpu);

    gpu_cl_enqueue_nd(ctx.gpu, ctx.exchange_id, 1, &ctx.local, &ctx.global, NULL);

    gpu_cl_fill_kernel_args(ctx.gpu, ctx.exchange_id, 0, 2, &ctx.min_gpu, sizeof(cl_mem), &ctx.g->m_buffer, sizeof(cl_mem));
    gpu_cl_enqueue_nd(ctx.gpu, ctx.exchange_id, 1, &ctx.local, &ctx.global, NULL);

    gsa_context_read_minimun_grid(&ctx);
    logging_log(LOG_INFO, "GSA Done. Minimun energy found %.15e eV", ctx.min_energy / QE);
}

void gsa(grid *g, gsa_params params) {
    gsa_base(g, params.qA, params.qV, params.qT, params.T0, params.inner_steps, params.outer_steps, params.print_factor, params.field_func, params.compile_augment);
}

void gsa_thermal_step(gsa_context *ctx) {
    int seed = rand();
    gpu_cl_set_kernel_arg(ctx->gpu, ctx->thermal_id, 6, sizeof(ctx->T), &ctx->T);
    gpu_cl_set_kernel_arg(ctx->gpu, ctx->thermal_id, 7, sizeof(seed), &seed);
    gpu_cl_enqueue_nd(ctx->gpu, ctx->thermal_id, 1, &ctx->local, &ctx->global, NULL);

    ctx->step++;
    ctx->inner_step++;
    ctx->T = ctx->Tqt / (pow(ctx->inner_step + 1.0, ctx->qT1) - 1.0);

    if (ctx->step % ctx->parameters.inner_steps == 0) {
        srand(time(NULL));
        ctx->outer_step++;
        ctx->inner_step = 0;
    }
}

void gsa_metropolis_step(gsa_context *ctx) {
    double new_energy = energy_from_gsa_context(ctx);

    if (new_energy <= ctx->min_energy) {
        ctx->min_energy = new_energy;
        gpu_cl_set_kernel_arg(ctx->gpu, ctx->exchange_id, 0, sizeof(cl_mem), &ctx->min_gpu);
        gpu_cl_enqueue_nd(ctx->gpu, ctx->exchange_id, 1, &ctx->local, &ctx->global, NULL);
    }

    if (new_energy <= ctx->last_energy) {
        ctx->last_energy = new_energy;
        gpu_cl_set_kernel_arg(ctx->gpu, ctx->exchange_id, 0, sizeof(cl_mem), &ctx->g->m_buffer);
        gpu_cl_enqueue_nd(ctx->gpu, ctx->exchange_id, 1, &ctx->local, &ctx->global, NULL);
    } else {
        double df = (new_energy - ctx->last_energy) / (ctx->g->gi.rows * ctx->g->gi.cols);// / fabs(ctx->g->gp->exchange);
        double pqa = 1.0 / pow(1.0 + ctx->qA1 * df / (KB * ctx->T), ctx->oneqA1);
        if (shit_random(0.0, 1.0) < pqa) {
            ctx->last_energy = new_energy;
            gpu_cl_set_kernel_arg(ctx->gpu, ctx->exchange_id, 0, sizeof(cl_mem), &ctx->g->m_buffer);
            gpu_cl_enqueue_nd(ctx->gpu, ctx->exchange_id, 1, &ctx->local, &ctx->global, NULL);
        }
    }

    if (ctx->inner_step % (ctx->parameters.inner_steps / ctx->parameters.print_factor) == 0)
        logging_log(LOG_INFO, "GSA Outer Step: %"PRIu64" GSA Inner Step: %"PRIu64" Minimun Energy %.15e eV Temperature: %.15e", ctx->outer_step, ctx->inner_step, ctx->min_energy / QE, ctx->T);
}

void gsa_context_close(gsa_context *ctx) {
    gpu_cl_read_buffer(ctx->gpu, ctx->g->gi.rows * ctx->g->gi.cols * sizeof(*ctx->g->m), 0, ctx->g->m, ctx->min_gpu);
    free(ctx->energy_cpu);
    gpu_cl_release_memory(ctx->swap_gpu);
    gpu_cl_release_memory(ctx->min_gpu);
    gpu_cl_release_memory(ctx->energy_gpu);
    memset(ctx, 0, sizeof(*ctx));
}

void gsa_context_read_minimun_grid(gsa_context *ctx) {
    gpu_cl_read_buffer(ctx->gpu, ctx->g->gi.rows * ctx->g->gi.cols * sizeof(*ctx->g->m), 0, ctx->g->m, ctx->min_gpu);
}

gsa_params gsa_params_init() {
    gsa_params ret = {0};
    ret.qA = 2.8;
    ret.qV = 2.6;
    ret.qT = 2.6;
    ret.T0 = 10.0;
    ret.inner_steps = 100000;
    ret.outer_steps = 15;
    ret.print_factor = 10;
    ret.field_func = str_is_cstr("return v3d_s(0);");
    ret.compile_augment = STR_NULL;
    return ret;
}
