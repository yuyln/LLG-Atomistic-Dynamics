#include "gradient_descent.h"
#include <inttypes.h>

static double energy_from_gradient_descent_context(gradient_descent_context *ctx) {
    gpu_cl_enqueue_nd(ctx->gpu, ctx->energy_id, 1, &ctx->local, &ctx->global, NULL);
    gpu_cl_read_buffer(ctx->gpu, ctx->g->gi.rows * ctx->g->gi.cols * sizeof(*ctx->energy_cpu), 0, ctx->energy_cpu, ctx->energy_gpu);
    double ret = 0.0;
    for (uint64_t i = 0; i < ctx->g->gi.rows * ctx->g->gi.cols; ++i)
        ret += ctx->energy_cpu[i];
    return ret;
}

gradient_descent_context gradient_descent_context_init(grid *g, gpu_cl *gpu, gradient_descent_params params) {
    gradient_descent_context ret = (gradient_descent_context){0};
    ret.g = g;
    ret.gpu = gpu;
    grid_to_gpu(ret.g, *ret.gpu);

    ret.global = g->gi.rows * g->gi.cols;
    ret.local = gpu_cl_gcd(ret.global, 32);
    ret.params = params;

    ret.before_gpu = gpu_cl_create_buffer(ret.gpu, sizeof(*g->m) * g->gi.cols * g->gi.rows, CL_MEM_READ_WRITE);
    ret.min_gpu = gpu_cl_create_buffer(ret.gpu, sizeof(*g->m) * g->gi.cols * g->gi.rows, CL_MEM_READ_WRITE);
    ret.after_gpu = gpu_cl_create_buffer(ret.gpu, sizeof(*g->m) * g->gi.cols * g->gi.rows, CL_MEM_READ_WRITE);
    ret.energy_gpu = gpu_cl_create_buffer(ret.gpu, sizeof(*ret.energy_cpu) * g->gi.cols * g->gi.rows, CL_MEM_READ_WRITE);

    ret.energy_cpu = calloc(g->gi.rows * g->gi.cols, sizeof(*ret.energy_cpu));
    if (!ret.energy_cpu)
        logging_log(LOG_FATAL, "Could not allocate[%"PRIu64" bytes] memory for energy on cpu %s", g->gi.rows * g->gi.cols * sizeof(*ret.energy_cpu), strerror(errno));


    ret.step_id = gpu_cl_append_kernel(ret.gpu, "gradient_descent_step");
    ret.exchange_id = gpu_cl_append_kernel(ret.gpu, "exchange_grid");
    ret.energy_id = gpu_cl_append_kernel(ret.gpu, "calculate_energy");

    gpu_cl_fill_kernel_args(gpu, ret.exchange_id, 0, 2, &ret.before_gpu, sizeof(cl_mem), &ret.g->m_buffer, sizeof(cl_mem));
    gpu_cl_enqueue_nd(gpu, ret.exchange_id, 1, &ret.local, &ret.global, NULL);

    gpu_cl_fill_kernel_args(gpu, ret.exchange_id, 0, 2, &ret.after_gpu, sizeof(cl_mem), &ret.g->m_buffer, sizeof(cl_mem));
    gpu_cl_enqueue_nd(gpu, ret.exchange_id, 1, &ret.local, &ret.global, NULL);

    gpu_cl_fill_kernel_args(gpu, ret.exchange_id, 0, 2, &ret.min_gpu, sizeof(cl_mem), &ret.g->m_buffer, sizeof(cl_mem));
    gpu_cl_enqueue_nd(gpu, ret.exchange_id, 1, &ret.local, &ret.global, NULL);


    double t = 0.0;
    gpu_cl_fill_kernel_args(gpu, ret.energy_id, 0, 5, &g->gp_buffer, sizeof(cl_mem), &g->m_buffer, sizeof(cl_mem), &g->gi, sizeof(g->gi), &ret.energy_gpu, sizeof(cl_mem), &t, sizeof(t));

    int seed = rand();
    gpu_cl_fill_kernel_args(gpu, ret.step_id, 0, 11, &g->gp_buffer, sizeof(cl_mem), &ret.before_gpu, sizeof(cl_mem), &g->m_buffer, sizeof(cl_mem), &ret.after_gpu, sizeof(cl_mem), &g->gi, sizeof(g->gi), &ret.params.mass, sizeof(ret.params.mass), &ret.params.T, sizeof(ret.params.T), &ret.params.damping, sizeof(ret.params.damping), &ret.params.restoring, sizeof(ret.params.restoring), &ret.params.dt, sizeof(ret.params.dt), &seed, sizeof(seed));

    ret.min_energy = energy_from_gradient_descent_context(&ret);

    return ret;
}

void gradient_descent_step(gradient_descent_context *ctx) {
    gpu_cl_set_kernel_arg(ctx->gpu, ctx->step_id, 6, sizeof(ctx->params.T), &ctx->params.T);
    int seed = rand();
    gpu_cl_set_kernel_arg(ctx->gpu, ctx->step_id, 10, sizeof(seed), &seed);
    gpu_cl_enqueue_nd(ctx->gpu, ctx->step_id, 1, &ctx->local, &ctx->global, NULL);
    ctx->params.T *= ctx->params.T_factor;
}

void gradient_descent_close(gradient_descent_context *ctx) {
    free(ctx->energy_cpu);
    gpu_cl_release_memory(ctx->before_gpu)
    gpu_cl_release_memory(ctx->after_gpu)
    gpu_cl_release_memory(ctx->min_gpu)
    gpu_cl_release_memory(ctx->energy_gpu)
    memset(ctx, 0, sizeof(*ctx));
}

void gradient_descent_exchange(gradient_descent_context *ctx) {
    gpu_cl_set_kernel_arg(ctx->gpu, ctx->exchange_id, 1, sizeof(cl_mem), &ctx->g->m_buffer);
    gpu_cl_set_kernel_arg(ctx->gpu, ctx->exchange_id, 0, sizeof(cl_mem), &ctx->before_gpu);
    gpu_cl_enqueue_nd(ctx->gpu, ctx->exchange_id, 1, &ctx->local, &ctx->global, NULL);

    gpu_cl_set_kernel_arg(ctx->gpu, ctx->exchange_id, 1, sizeof(cl_mem), &ctx->after_gpu);
    gpu_cl_set_kernel_arg(ctx->gpu, ctx->exchange_id, 0, sizeof(cl_mem), &ctx->g->m_buffer);
    gpu_cl_enqueue_nd(ctx->gpu, ctx->exchange_id, 1, &ctx->local, &ctx->global, NULL);

    double new_energy = energy_from_gradient_descent_context(ctx);
    if (new_energy <= ctx->min_energy) {
        ctx->min_energy = new_energy;

        gpu_cl_set_kernel_arg(ctx->gpu, ctx->exchange_id, 1, sizeof(cl_mem), &ctx->g->m_buffer);
        gpu_cl_set_kernel_arg(ctx->gpu, ctx->exchange_id, 0, sizeof(cl_mem), &ctx->min_gpu);
        gpu_cl_enqueue_nd(ctx->gpu, ctx->exchange_id, 1, &ctx->local, &ctx->global, NULL);
    }
}

void gradient_descent_read_mininum_grid(gradient_descent_context *ctx) {
    v3d_from_gpu(ctx->g->m, ctx->min_gpu, ctx->g->gi.rows, ctx->g->gi.cols, *ctx->gpu);
}

gradient_descent_params gradient_descent_params_init() {
    gradient_descent_params ret = {0};
    ret.T = 100.0;
    ret.mass = 1.0;
    ret.dt = 1.0e-3;
    ret.damping = 0.0;
    ret.restoring = 0.0;
    ret.T_factor = 0.999999;
    ret.field_func = str_is_cstr("return v3d_s(0);");
    ret.compile_augment = STR_NULL;
    return ret;
}
