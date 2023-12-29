#include "grid_render.h"
#include "complete_kernel.h"
#include "constants.h"
#include <float.h>

grid_renderer grid_renderer_init(grid *g, render_window *window, string_view current_generation_func, string_view field_generation_func, string_view kernel_augment, string_view compile_augment) {
    UNUSED(kernel_augment);
    UNUSED(compile_augment);
    UNUSED(current_generation_func);
    UNUSED(field_generation_func);
    grid_renderer ret = {0};
    ret.width = window_width(window);
    ret.height = window_height(window);
    ret.window = window;
    ret.g = g;
    ret.gpu = gpu_cl_init(0, 0);
    grid_to_gpu(g, ret.gpu);

    cl_int err;

    ret.rgba_cpu = calloc(ret.width * ret.height, sizeof(*ret.rgba_cpu));
    ret.rgba_gpu = clCreateBuffer(ret.gpu.ctx, CL_MEM_READ_WRITE, sizeof(*ret.rgba_cpu) * ret.width * ret.height, NULL, &err);
    clw_print_cl_error(stderr, err, "[ FATAL ] Could not create RGBA buffer on gpu");

    ret.buffer_cpu = calloc(ret.width * ret.height, sizeof(*ret.buffer_cpu));
    ret.buffer_gpu = clCreateBuffer(ret.gpu.ctx, CL_MEM_READ_WRITE, sizeof(*ret.buffer_cpu) * ret.width * ret.height, NULL, &err);
    clw_print_cl_error(stderr, err, "[ FATAL ] Could not create buffer buffer on gpu");


    const char cmp[] = "-DOPENCL_COMPILATION";
    gpu_cl_compile_source(&ret.gpu, sv_from_cstr(complete_kernel), sv_from_cstr(cmp));


    ret.grid_hsl_id = gpu_append_kernel(&ret.gpu, "render_grid_hsl");
    ret.grid_bwr_id = gpu_append_kernel(&ret.gpu, "render_grid_bwr");
    ret.energy_id = gpu_append_kernel(&ret.gpu, "render_energy");
    ret.charge_id = gpu_append_kernel(&ret.gpu, "render_charge");
    ret.calc_charge_id = gpu_append_kernel(&ret.gpu, "calculate_charge_to_render");
    ret.calc_energy_id = gpu_append_kernel(&ret.gpu, "calculate_energy_to_render");

    /*clw_set_kernel_arg(ret.gpu.kernels[ret.grid_bwr_id], 0, sizeof(cl_mem), &ret.g->m_buffer);
    clw_set_kernel_arg(ret.gpu.kernels[ret.grid_bwr_id], 1, sizeof(ret.g->gi), &ret.g->gi);
    clw_set_kernel_arg(ret.gpu.kernels[ret.grid_bwr_id], 2, sizeof(cl_mem), &ret.rgba_gpu);
    clw_set_kernel_arg(ret.gpu.kernels[ret.grid_bwr_id], 3, sizeof(ret.width), &ret.width);
    clw_set_kernel_arg(ret.gpu.kernels[ret.grid_bwr_id], 4, sizeof(ret.height), &ret.height);*/

    gpu_fill_kernel_args(&ret.gpu, ret.grid_hsl_id, 0, 5, &ret.g->m_buffer, sizeof(cl_mem), &ret.g->gi, sizeof(ret.g->gi), &ret.rgba_gpu, sizeof(cl_mem), &ret.width, sizeof(ret.width), &ret.height, sizeof(ret.height));

    gpu_fill_kernel_args(&ret.gpu, ret.grid_bwr_id, 0, 5, &ret.g->m_buffer, sizeof(cl_mem), &ret.g->gi, sizeof(ret.g->gi), &ret.rgba_gpu, sizeof(cl_mem), &ret.width, sizeof(ret.width), &ret.height, sizeof(ret.height));

    gpu_fill_kernel_args(&ret.gpu, ret.calc_charge_id, 0, 3, &ret.g->m_buffer, sizeof(cl_mem), &ret.g->gi, sizeof(ret.g->gi), &ret.buffer_gpu, sizeof(cl_mem));

    //Need to set time
    gpu_fill_kernel_args(&ret.gpu, ret.calc_energy_id, 0, 4, &ret.g->gp_buffer, sizeof(cl_mem), &ret.g->m_buffer, sizeof(cl_mem), &ret.g->gi, sizeof(ret.g->gi), &ret.buffer_gpu, sizeof(cl_mem));


    gpu_fill_kernel_args(&ret.gpu, ret.charge_id, 0, 3, &ret.buffer_gpu, sizeof(cl_mem), &ret.g->gi.rows, sizeof(ret.g->gi.rows), &ret.g->gi.cols, sizeof(ret.g->gi.cols));
    gpu_fill_kernel_args(&ret.gpu, ret.charge_id, 5, 3, &ret.rgba_gpu, sizeof(cl_mem), &ret.width, sizeof(ret.width), &ret.height, sizeof(ret.height)); //Need to set ranges


    gpu_fill_kernel_args(&ret.gpu, ret.energy_id, 0, 3, &ret.buffer_gpu, sizeof(cl_mem), &ret.g->gi.rows, sizeof(ret.g->gi.rows), &ret.g->gi.cols, sizeof(ret.g->gi.cols));
    gpu_fill_kernel_args(&ret.gpu, ret.energy_id, 5, 3, &ret.rgba_gpu, sizeof(cl_mem), &ret.width, sizeof(ret.width), &ret.height, sizeof(ret.height)); //Need to set ranges

    return ret;
}

void grid_renderer_close(grid_renderer *gr) {
    free(gr->rgba_cpu);
    free(gr->buffer_cpu);
    clw_print_cl_error(stderr, clReleaseMemObject(gr->rgba_gpu), "[ FATAL ] Could not release RGBA buffer from gpu");
    clw_print_cl_error(stderr, clReleaseMemObject(gr->buffer_gpu), "[ FATAL ] Could not release buffer buffer from gpu");
    grid_release_from_gpu(gr->g);
    gpu_cl_close(&gr->gpu);
}

void grid_renderer_hsl(grid_renderer *gr) {
    size_t global = gr->width * gr->height;
    size_t local = clw_gcd(global, 32);
    clw_enqueue_nd(gr->gpu.queue, gr->gpu.kernels[gr->grid_hsl_id], 1, NULL, &global, &local);
    clw_print_cl_error(stderr, clEnqueueReadBuffer(gr->gpu.queue, gr->rgba_gpu, CL_TRUE, 0, gr->width * gr->height * sizeof(*gr->rgba_cpu), gr->rgba_cpu, 0, NULL, NULL), "[ FATAL ] Could not read RGBA buffer from GPU");
    window_draw_from_bytes(gr->window, gr->rgba_cpu, 0, 0, gr->width, gr->height);
}

void grid_renderer_bwr(grid_renderer *gr) {
    size_t global = gr->width * gr->height;
    size_t local = clw_gcd(global, 32);
    clw_enqueue_nd(gr->gpu.queue, gr->gpu.kernels[gr->grid_bwr_id], 1, NULL, &global, &local);
    clw_print_cl_error(stderr, clEnqueueReadBuffer(gr->gpu.queue, gr->rgba_gpu, CL_TRUE, 0, gr->width * gr->height * sizeof(*gr->rgba_cpu), gr->rgba_cpu, 0, NULL, NULL), "[ FATAL ] Could not read RGBA buffer from GPU");
    window_draw_from_bytes(gr->window, gr->rgba_cpu, 0, 0, gr->width, gr->height);
}

void grid_renderer_energy(grid_renderer *gr, double time) {
    size_t global = gr->g->gi.rows * gr->g->gi.cols;
    size_t local = clw_gcd(global, 32);

    clw_set_kernel_arg(gr->gpu.kernels[gr->calc_energy_id], 4, sizeof(double), &time);
    clw_enqueue_nd(gr->gpu.queue, gr->gpu.kernels[gr->calc_energy_id], 1, NULL, &global, &local);


    clw_print_cl_error(stderr, clEnqueueReadBuffer(gr->gpu.queue, gr->buffer_gpu, CL_TRUE, 0, global * sizeof(*gr->buffer_cpu), gr->buffer_cpu, 0, NULL, NULL), "[ FATAL ] Could not read RGBA buffer from GPU");

    double min_energy = FLT_MAX;
    double max_energy = -FLT_MAX;

    for (uint64_t i = 0; i < global; ++i) {
        if (gr->buffer_cpu[i] < min_energy) min_energy = gr->buffer_cpu[i];
        if (gr->buffer_cpu[i] > max_energy) max_energy = gr->buffer_cpu[i];
    }

    global = gr->width * gr->height;
    local = clw_gcd(global, 32);

    clw_set_kernel_arg(gr->gpu.kernels[gr->energy_id], 3, sizeof(double), &min_energy);
    clw_set_kernel_arg(gr->gpu.kernels[gr->energy_id], 4, sizeof(double), &max_energy);
    clw_enqueue_nd(gr->gpu.queue, gr->gpu.kernels[gr->energy_id], 1, NULL, &global, &local);

    clw_print_cl_error(stderr, clEnqueueReadBuffer(gr->gpu.queue, gr->rgba_gpu, CL_TRUE, 0, gr->width * gr->height * sizeof(*gr->rgba_cpu), gr->rgba_cpu, 0, NULL, NULL), "[ FATAL ] Could not read RGBA buffer from GPU");
    window_draw_from_bytes(gr->window, gr->rgba_cpu, 0, 0, gr->width, gr->height);
}

void grid_renderer_charge(grid_renderer *gr) {
    size_t global = gr->g->gi.rows * gr->g->gi.cols;
    size_t local = clw_gcd(global, 32);

    clw_enqueue_nd(gr->gpu.queue, gr->gpu.kernels[gr->calc_charge_id], 1, NULL, &global, &local);

    clw_print_cl_error(stderr, clEnqueueReadBuffer(gr->gpu.queue, gr->buffer_gpu, CL_TRUE, 0, global * sizeof(*gr->buffer_cpu), gr->buffer_cpu, 0, NULL, NULL), "[ FATAL ] Could not read RGBA buffer from GPU");

    double min_charge = FLT_MAX;
    double max_charge = -FLT_MAX;

    for (uint64_t i = 0; i < global; ++i) {
        if (gr->buffer_cpu[i] < min_charge) min_charge = gr->buffer_cpu[i];
        if (gr->buffer_cpu[i] > max_charge) max_charge = gr->buffer_cpu[i];
    }

    global = gr->width * gr->height;
    local = clw_gcd(global, 32);

    clw_set_kernel_arg(gr->gpu.kernels[gr->charge_id], 3, sizeof(double), &min_charge);
    clw_set_kernel_arg(gr->gpu.kernels[gr->charge_id], 4, sizeof(double), &max_charge);
    clw_enqueue_nd(gr->gpu.queue, gr->gpu.kernels[gr->charge_id], 1, NULL, &global, &local);

    clw_print_cl_error(stderr, clEnqueueReadBuffer(gr->gpu.queue, gr->rgba_gpu, CL_TRUE, 0, gr->width * gr->height * sizeof(*gr->rgba_cpu), gr->rgba_cpu, 0, NULL, NULL), "[ FATAL ] Could not read RGBA buffer from GPU");
    window_draw_from_bytes(gr->window, gr->rgba_cpu, 0, 0, gr->width, gr->height);
}
