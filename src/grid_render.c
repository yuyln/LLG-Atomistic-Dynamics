#include "grid_render.h"
#include "complete_kernel.h"
#include "constants.h"
#include "kernel_funcs.h"
#include <float.h>

grid_renderer grid_renderer_init(grid *g, gpu_cl *gpu) {
    grid_renderer ret = {0};
    ret.width = window_width();
    ret.height = window_height();
    ret.g = g;
    ret.gpu = gpu;
    grid_to_gpu(g, *ret.gpu);

    cl_int err;

    ret.rgba_cpu = calloc(ret.width * ret.height, sizeof(*ret.rgba_cpu));
    ret.rgba_gpu = clCreateBuffer(ret.gpu->ctx, CL_MEM_READ_WRITE, sizeof(*ret.rgba_cpu) * ret.width * ret.height, NULL, &err);
    clw_print_cl_error(stderr, err, "[ FATAL ] Could not create RGBA buffer on gpu");

    ret.buffer_cpu = calloc(ret.width * ret.height, sizeof(*ret.buffer_cpu));
    ret.buffer_gpu = clCreateBuffer(ret.gpu->ctx, CL_MEM_READ_WRITE, sizeof(*ret.buffer_cpu) * ret.width * ret.height, NULL, &err);
    clw_print_cl_error(stderr, err, "[ FATAL ] Could not create buffer buffer on gpu");

    ret.grid_hsl_id = gpu_append_kernel(ret.gpu, "render_grid_hsl");
    ret.grid_bwr_id = gpu_append_kernel(ret.gpu, "render_grid_bwr");
    ret.energy_id = gpu_append_kernel(ret.gpu, "render_energy");
    ret.charge_id = gpu_append_kernel(ret.gpu, "render_charge");
    ret.calc_charge_id = gpu_append_kernel(ret.gpu, "calculate_charge_to_render");
    ret.calc_energy_id = gpu_append_kernel(ret.gpu, "calculate_energy");

    gpu_fill_kernel_args(ret.gpu, ret.grid_hsl_id, 0, 5, &ret.g->m_buffer, sizeof(cl_mem), &ret.g->gi, sizeof(ret.g->gi), &ret.rgba_gpu, sizeof(cl_mem), &ret.width, sizeof(ret.width), &ret.height, sizeof(ret.height));

    gpu_fill_kernel_args(ret.gpu, ret.grid_bwr_id, 0, 5, &ret.g->m_buffer, sizeof(cl_mem), &ret.g->gi, sizeof(ret.g->gi), &ret.rgba_gpu, sizeof(cl_mem), &ret.width, sizeof(ret.width), &ret.height, sizeof(ret.height));

    gpu_fill_kernel_args(ret.gpu, ret.calc_charge_id, 0, 3, &ret.g->m_buffer, sizeof(cl_mem), &ret.g->gi, sizeof(ret.g->gi), &ret.buffer_gpu, sizeof(cl_mem));

    //Need to set time
    gpu_fill_kernel_args(ret.gpu, ret.calc_energy_id, 0, 4, &ret.g->gp_buffer, sizeof(cl_mem), &ret.g->m_buffer, sizeof(cl_mem), &ret.g->gi, sizeof(ret.g->gi), &ret.buffer_gpu, sizeof(cl_mem));


    gpu_fill_kernel_args(ret.gpu, ret.charge_id, 0, 3, &ret.buffer_gpu, sizeof(cl_mem), &ret.g->gi.rows, sizeof(ret.g->gi.rows), &ret.g->gi.cols, sizeof(ret.g->gi.cols));
    gpu_fill_kernel_args(ret.gpu, ret.charge_id, 5, 3, &ret.rgba_gpu, sizeof(cl_mem), &ret.width, sizeof(ret.width), &ret.height, sizeof(ret.height)); //Need to set ranges

    gpu_fill_kernel_args(ret.gpu, ret.energy_id, 0, 3, &ret.buffer_gpu, sizeof(cl_mem), &ret.g->gi.rows, sizeof(ret.g->gi.rows), &ret.g->gi.cols, sizeof(ret.g->gi.cols));
    gpu_fill_kernel_args(ret.gpu, ret.energy_id, 5, 3, &ret.rgba_gpu, sizeof(cl_mem), &ret.width, sizeof(ret.width), &ret.height, sizeof(ret.height)); //Need to set ranges

    return ret;
}

void grid_renderer_close(grid_renderer *gr) {
    free(gr->rgba_cpu);
    free(gr->buffer_cpu);
    clw_print_cl_error(stderr, clReleaseMemObject(gr->rgba_gpu), "[ FATAL ] Could not release RGBA buffer from gpu");
    clw_print_cl_error(stderr, clReleaseMemObject(gr->buffer_gpu), "[ FATAL ] Could not release buffer buffer from gpu");
    grid_release_from_gpu(gr->g);
}

void grid_renderer_hsl(grid_renderer *gr) {
    size_t global = gr->width * gr->height;
    size_t local = clw_gcd(global, 32);
    cl_event render_grid = clw_enqueue_nd(gr->gpu->queue, gr->gpu->kernels[gr->grid_hsl_id], 1, NULL, &global, &local);
    clw_print_cl_error(stderr, clEnqueueReadBuffer(gr->gpu->queue, gr->rgba_gpu, CL_TRUE, 0, gr->width * gr->height * sizeof(*gr->rgba_cpu), gr->rgba_cpu, 0, NULL, NULL), "[ FATAL ] Could not read RGBA buffer from GPU");
    window_draw_from_bytes(gr->rgba_cpu, 0, 0, gr->width, gr->height);

    gpu_profiling(stdout, render_grid, "Grid HSL Render");
}

void grid_renderer_bwr(grid_renderer *gr) {
    size_t global = gr->width * gr->height;
    size_t local = clw_gcd(global, 32);
    cl_event render_grid = clw_enqueue_nd(gr->gpu->queue, gr->gpu->kernels[gr->grid_bwr_id], 1, NULL, &global, &local);
    clw_print_cl_error(stderr, clEnqueueReadBuffer(gr->gpu->queue, gr->rgba_gpu, CL_TRUE, 0, gr->width * gr->height * sizeof(*gr->rgba_cpu), gr->rgba_cpu, 0, NULL, NULL), "[ FATAL ] Could not read RGBA buffer from GPU");
    window_draw_from_bytes(gr->rgba_cpu, 0, 0, gr->width, gr->height);

    gpu_profiling(stdout, render_grid, "Grid BWR Render");
}

void grid_renderer_energy(grid_renderer *gr, double time) {
    size_t global = gr->g->gi.rows * gr->g->gi.cols;
    size_t local = clw_gcd(global, 32);

    clw_set_kernel_arg(gr->gpu->kernels[gr->calc_energy_id], 4, sizeof(double), &time);
    cl_event calc_energy = clw_enqueue_nd(gr->gpu->queue, gr->gpu->kernels[gr->calc_energy_id], 1, NULL, &global, &local);

    clw_print_cl_error(stderr, clEnqueueReadBuffer(gr->gpu->queue, gr->buffer_gpu, CL_TRUE, 0, global * sizeof(*gr->buffer_cpu), gr->buffer_cpu, 0, NULL, NULL), "[ FATAL ] Could not read RGBA buffer from GPU");

    double min_energy = FLT_MAX;
    double max_energy = -FLT_MAX;

    for (uint64_t i = 0; i < global; ++i) {
        if (gr->buffer_cpu[i] < min_energy) min_energy = gr->buffer_cpu[i];
        if (gr->buffer_cpu[i] > max_energy) max_energy = gr->buffer_cpu[i];
    }

    global = gr->width * gr->height;
    local = clw_gcd(global, 32);
    clw_set_kernel_arg(gr->gpu->kernels[gr->energy_id], 3, sizeof(double), &min_energy);
    clw_set_kernel_arg(gr->gpu->kernels[gr->energy_id], 4, sizeof(double), &max_energy);

    cl_event render_energy = clw_enqueue_nd(gr->gpu->queue, gr->gpu->kernels[gr->energy_id], 1, NULL, &global, &local);

    clw_print_cl_error(stderr, clEnqueueReadBuffer(gr->gpu->queue, gr->rgba_gpu, CL_TRUE, 0, gr->width * gr->height * sizeof(*gr->rgba_cpu), gr->rgba_cpu, 0, NULL, NULL), "[ FATAL ] Could not read RGBA buffer from GPU");
    window_draw_from_bytes(gr->rgba_cpu, 0, 0, gr->width, gr->height);

    gpu_profiling(stdout, calc_energy, "Energy Calculation");
    gpu_profiling(stdout, render_energy, "Energy Rendering");
}

void grid_renderer_charge(grid_renderer *gr) {
    size_t global = gr->g->gi.rows * gr->g->gi.cols;
    size_t local = clw_gcd(global, 32);

    cl_event calc_charge = clw_enqueue_nd(gr->gpu->queue, gr->gpu->kernels[gr->calc_charge_id], 1, NULL, &global, &local);

    clw_print_cl_error(stderr, clEnqueueReadBuffer(gr->gpu->queue, gr->buffer_gpu, CL_TRUE, 0, global * sizeof(*gr->buffer_cpu), gr->buffer_cpu, 0, NULL, NULL), "[ FATAL ] Could not read RGBA buffer from GPU");

    double min_charge = FLT_MAX;
    double max_charge = -FLT_MAX;

    for (uint64_t i = 0; i < global; ++i) {
        if (gr->buffer_cpu[i] < min_charge) min_charge = gr->buffer_cpu[i];
        if (gr->buffer_cpu[i] > max_charge) max_charge = gr->buffer_cpu[i];
    }

    global = gr->width * gr->height;
    local = clw_gcd(global, 32);

    clw_set_kernel_arg(gr->gpu->kernels[gr->charge_id], 3, sizeof(double), &min_charge);
    clw_set_kernel_arg(gr->gpu->kernels[gr->charge_id], 4, sizeof(double), &max_charge);

    cl_event render_charge = clw_enqueue_nd(gr->gpu->queue, gr->gpu->kernels[gr->charge_id], 1, NULL, &global, &local);

    clw_print_cl_error(stderr, clEnqueueReadBuffer(gr->gpu->queue, gr->rgba_gpu, CL_TRUE, 0, gr->width * gr->height * sizeof(*gr->rgba_cpu), gr->rgba_cpu, 0, NULL, NULL), "[ FATAL ] Could not read RGBA buffer from GPU");
    window_draw_from_bytes(gr->rgba_cpu, 0, 0, gr->width, gr->height);

    gpu_profiling(stdout, calc_charge, "Charge Calculation");
    gpu_profiling(stdout, render_charge, "Charge Rendering  ");
}
