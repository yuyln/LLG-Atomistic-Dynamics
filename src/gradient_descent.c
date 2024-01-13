#include "gradient_descent.h"
static double energy_from_gradient_descent_context(gradient_descent_context *ctx) {
    clw_enqueue_nd(ctx->gpu->queue, ctx->gpu->kernels[ctx->energy_id], 1, NULL, &ctx->global, &ctx->local);
    clw_print_cl_error(stderr, clEnqueueReadBuffer(ctx->gpu->queue, ctx->energy_gpu, CL_TRUE, 0, ctx->g->gi.rows * ctx->g->gi.cols * sizeof(*ctx->energy_cpu), ctx->energy_cpu, 0, NULL, NULL), "[ FATAL ] Could not read energy buffer gradient descent");
    double ret = 0.0;
    for (uint64_t i = 0; i < ctx->g->gi.rows * ctx->g->gi.cols; ++i)
        ret += ctx->energy_cpu[i];
    return ret;
}

gradient_descent_context gradient_descent_context_init_params(grid *g, gpu_cl *gpu, gradient_descent_param param) {
    return gradient_descent_context_init_base(g, gpu, param.T, param.mass, param.dt, param.damping, param.restoring, param.T_factor);
}

gradient_descent_context gradient_descent_context_init_base(grid *g, gpu_cl *gpu, double T, double mass, double dt, double damping, double restoring, double T_factor) {
    gradient_descent_context ret = (gradient_descent_context){0};
    ret.g = g;
    ret.gpu = gpu;
    grid_to_gpu(ret.g, *ret.gpu);

    ret.global = g->gi.rows * g->gi.cols;
    ret.local = clw_gcd(ret.global, 32);
    ret.T = T;
    ret.mass = mass;
    ret.dt = dt;
    ret.damping = damping;
    ret.restoring = restoring;
    ret.T_factor = T_factor;

    cl_int err;
    ret.before_gpu = clCreateBuffer(ret.gpu->ctx, CL_MEM_READ_WRITE, sizeof(*g->m) * g->gi.cols * g->gi.rows, NULL, &err);
    clw_print_cl_error(stderr, err, "[ FATAL ] Could not create buffer for old_grid gradient descent");

    ret.min_gpu = clCreateBuffer(ret.gpu->ctx, CL_MEM_READ_WRITE, sizeof(*g->m) * g->gi.cols * g->gi.rows, NULL, &err);
    clw_print_cl_error(stderr, err, "[ FATAL ] Could not create buffer for min_grid gradient descent");

    ret.after_gpu = clCreateBuffer(ret.gpu->ctx, CL_MEM_READ_WRITE, sizeof(*g->m) * g->gi.cols * g->gi.rows, NULL, &err);
    clw_print_cl_error(stderr, err, "[ FATAL ] Could not create buffer for after_grid gradient descent");

    ret.energy_cpu = calloc(g->gi.rows * g->gi.cols, sizeof(*ret.energy_cpu));
    ret.energy_gpu = clCreateBuffer(ret.gpu->ctx, CL_MEM_READ_WRITE, sizeof(*ret.energy_cpu) * g->gi.cols * g->gi.rows, NULL, &err);
    clw_print_cl_error(stderr, err, "[ FATAL ] Could not create buffer for energy_gpu gradient descent");


    ret.step_id = gpu_append_kernel(ret.gpu, "gradient_descent_step");
    ret.exchange_id = gpu_append_kernel(ret.gpu, "exchange_grid");
    ret.energy_id = gpu_append_kernel(ret.gpu, "calculate_energy");

    gpu_fill_kernel_args(gpu, ret.exchange_id, 0, 2, &ret.before_gpu, sizeof(cl_mem), &ret.g->m_buffer, sizeof(cl_mem));
    clw_enqueue_nd(ret.gpu->queue, ret.gpu->kernels[ret.exchange_id], 1, NULL, &ret.global, &ret.local);

    gpu_fill_kernel_args(gpu, ret.exchange_id, 0, 2, &ret.after_gpu, sizeof(cl_mem), &ret.g->m_buffer, sizeof(cl_mem));
    clw_enqueue_nd(ret.gpu->queue, ret.gpu->kernels[ret.exchange_id], 1, NULL, &ret.global, &ret.local);

    gpu_fill_kernel_args(gpu, ret.exchange_id, 0, 2, &ret.min_gpu, sizeof(cl_mem), &ret.g->m_buffer, sizeof(cl_mem));
    clw_enqueue_nd(ret.gpu->queue, ret.gpu->kernels[ret.exchange_id], 1, NULL, &ret.global, &ret.local);


    double t = 0.0;
    gpu_fill_kernel_args(gpu, ret.energy_id, 0, 5, &g->gp_buffer, sizeof(cl_mem), &g->m_buffer, sizeof(cl_mem), &g->gi, sizeof(g->gi), &ret.energy_gpu, sizeof(cl_mem), &t, sizeof(t));

    int seed = rand();
    gpu_fill_kernel_args(gpu, ret.step_id, 0, 11, &g->gp_buffer, sizeof(cl_mem), &ret.before_gpu, sizeof(cl_mem), &g->m_buffer, sizeof(cl_mem), &ret.after_gpu, sizeof(cl_mem), &g->gi, sizeof(g->gi), &ret.mass, sizeof(ret.mass), &ret.T, sizeof(ret.T), &ret.damping, sizeof(ret.damping), &ret.restoring, sizeof(ret.restoring), &ret.dt, sizeof(ret.dt), &seed, sizeof(seed));

    ret.min_energy = energy_from_gradient_descent_context(&ret);

    return ret;
}

void gradient_descent_step(gradient_descent_context *ctx) {
    clw_set_kernel_arg(ctx->gpu->kernels[ctx->step_id], 6, sizeof(ctx->T), &ctx->T);
    int seed = rand();
    clw_set_kernel_arg(ctx->gpu->kernels[ctx->step_id], 10, sizeof(seed), &seed);
    clw_enqueue_nd(ctx->gpu->queue, ctx->gpu->kernels[ctx->step_id], 1, NULL, &ctx->global, &ctx->local);
    ctx->T *= ctx->T_factor;
}

void gradient_descent_clear(gradient_descent_context *ctx) {
    free(ctx->energy_cpu);
    clw_print_cl_error(stderr, clReleaseMemObject(ctx->before_gpu), "[ FATAL ] Could not release before_buffer from GPU gsa");
    clw_print_cl_error(stderr, clReleaseMemObject(ctx->after_gpu), "[ FATAL ] Could not release after_buffer from GPU gsa");
    clw_print_cl_error(stderr, clReleaseMemObject(ctx->min_gpu), "[ FATAL ] Could not release min_gpu from GPU gsa");
    clw_print_cl_error(stderr, clReleaseMemObject(ctx->energy_gpu), "[ FATAL ] Could not release energy_buffer from GPU gsa");
    memset(ctx, 0, sizeof(*ctx));
}

void gradient_descent_exchange(gradient_descent_context *ctx) {
    clw_print_cl_error(stderr, clSetKernelArg(ctx->gpu->kernels[ctx->exchange_id].kernel, 1, sizeof(cl_mem), &ctx->g->m_buffer), "[ FATAL ] Could not set current grid as argument of exchange grids GSA");

    clw_print_cl_error(stderr, clSetKernelArg(ctx->gpu->kernels[ctx->exchange_id].kernel, 0, sizeof(cl_mem), &ctx->before_gpu), "[ FATAL ] Could not set before grid as argument of exchange grids GSA");

    clw_enqueue_nd(ctx->gpu->queue, ctx->gpu->kernels[ctx->exchange_id], 1, NULL, &ctx->global, &ctx->local);

    clw_print_cl_error(stderr, clSetKernelArg(ctx->gpu->kernels[ctx->exchange_id].kernel, 1, sizeof(cl_mem), &ctx->after_gpu), "[ FATAL ] Could not set after grid as argument of exchange grids GSA");

    clw_print_cl_error(stderr, clSetKernelArg(ctx->gpu->kernels[ctx->exchange_id].kernel, 0, sizeof(cl_mem), &ctx->g->m_buffer), "[ FATAL ] Could not set current grid as argument of exchange grids GSA");

    clw_enqueue_nd(ctx->gpu->queue, ctx->gpu->kernels[ctx->exchange_id], 1, NULL, &ctx->global, &ctx->local);

    double new_energy = energy_from_gradient_descent_context(ctx);
    if (new_energy <= ctx->min_energy) {
        ctx->min_energy = new_energy;
        clw_print_cl_error(stderr, clSetKernelArg(ctx->gpu->kernels[ctx->exchange_id].kernel, 1, sizeof(cl_mem), &ctx->g->m_buffer), "[ FATAL ] Could not set current grid as argument of exchange grids GSA");

        clw_print_cl_error(stderr, clSetKernelArg(ctx->gpu->kernels[ctx->exchange_id].kernel, 0, sizeof(cl_mem), &ctx->min_gpu), "[ FATAL ] Could not set min grid as argument of exchange grids GSA");

        clw_enqueue_nd(ctx->gpu->queue, ctx->gpu->kernels[ctx->exchange_id], 1, NULL, &ctx->global, &ctx->local);
    }
}

void gradient_descente_read_mininum_grid(gradient_descent_context *ctx) {
    v3d_from_gpu(ctx->g->m, ctx->min_gpu, ctx->g->gi.rows, ctx->g->gi.cols, *ctx->gpu);
}
