#include "grid_render.h"
#include "complete_kernel.h"
#include "constants.h"
#include "kernel_funcs.h"
#include "gsa.h"
#include "integrate.h"
#include "gradient_descent.h"
#include <float.h>
#include <inttypes.h>

grid_renderer grid_renderer_init(grid *g, gpu_cl *gpu) {
    grid_renderer ret = {0};
    ret.width = window_width();
    ret.height = window_height();
    ret.g = g;
    ret.gpu = gpu;
    grid_to_gpu(g, *ret.gpu);

    ret.rgba_cpu = calloc(ret.width * ret.height, sizeof(*ret.rgba_cpu));
    ret.rgba_gpu = gpu_cl_create_buffer(ret.gpu, sizeof(*ret.rgba_cpu) * ret.width * ret.height, CL_MEM_READ_WRITE);

    ret.buffer_cpu = calloc(ret.width * ret.height, sizeof(*ret.buffer_cpu));
    ret.buffer_gpu = gpu_cl_create_buffer(ret.gpu, sizeof(*ret.buffer_cpu) * ret.width * ret.height, CL_MEM_READ_WRITE);

    ret.grid_hsl_id = gpu_cl_append_kernel(ret.gpu, "render_grid_hsl");
    ret.grid_bwr_id = gpu_cl_append_kernel(ret.gpu, "render_grid_bwr");
    ret.energy_id = gpu_cl_append_kernel(ret.gpu, "render_energy");
    ret.charge_id = gpu_cl_append_kernel(ret.gpu, "render_charge");
    ret.calc_charge_id = gpu_cl_append_kernel(ret.gpu, "calculate_charge_to_render");
    ret.calc_energy_id = gpu_cl_append_kernel(ret.gpu, "calculate_energy");

    gpu_cl_fill_kernel_args(ret.gpu, ret.grid_hsl_id, 0, 5, &ret.g->m_buffer, sizeof(cl_mem), &ret.g->gi, sizeof(ret.g->gi), &ret.rgba_gpu, sizeof(cl_mem), &ret.width, sizeof(ret.width), &ret.height, sizeof(ret.height));

    gpu_cl_fill_kernel_args(ret.gpu, ret.grid_bwr_id, 0, 5, &ret.g->m_buffer, sizeof(cl_mem), &ret.g->gi, sizeof(ret.g->gi), &ret.rgba_gpu, sizeof(cl_mem), &ret.width, sizeof(ret.width), &ret.height, sizeof(ret.height));

    gpu_cl_fill_kernel_args(ret.gpu, ret.calc_charge_id, 0, 3, &ret.g->m_buffer, sizeof(cl_mem), &ret.g->gi, sizeof(ret.g->gi), &ret.buffer_gpu, sizeof(cl_mem));

    //Need to set time
    gpu_cl_fill_kernel_args(ret.gpu, ret.calc_energy_id, 0, 4, &ret.g->gp_buffer, sizeof(cl_mem), &ret.g->m_buffer, sizeof(cl_mem), &ret.g->gi, sizeof(ret.g->gi), &ret.buffer_gpu, sizeof(cl_mem));


    gpu_cl_fill_kernel_args(ret.gpu, ret.charge_id, 0, 3, &ret.buffer_gpu, sizeof(cl_mem), &ret.g->gi.rows, sizeof(ret.g->gi.rows), &ret.g->gi.cols, sizeof(ret.g->gi.cols));
    gpu_cl_fill_kernel_args(ret.gpu, ret.charge_id, 5, 3, &ret.rgba_gpu, sizeof(cl_mem), &ret.width, sizeof(ret.width), &ret.height, sizeof(ret.height)); //Need to set ranges

    gpu_cl_fill_kernel_args(ret.gpu, ret.energy_id, 0, 3, &ret.buffer_gpu, sizeof(cl_mem), &ret.g->gi.rows, sizeof(ret.g->gi.rows), &ret.g->gi.cols, sizeof(ret.g->gi.cols));
    gpu_cl_fill_kernel_args(ret.gpu, ret.energy_id, 5, 3, &ret.rgba_gpu, sizeof(cl_mem), &ret.width, sizeof(ret.width), &ret.height, sizeof(ret.height)); //Need to set ranges

    return ret;
}

void grid_renderer_close(grid_renderer *gr) {
    free(gr->rgba_cpu);
    free(gr->buffer_cpu);
    gpu_cl_release_memory(gr->rgba_gpu);
    gpu_cl_release_memory(gr->buffer_gpu);
    grid_release_from_gpu(gr->g);
}

void grid_renderer_hsl(grid_renderer *gr) {
    size_t global = gr->width * gr->height;
    size_t local = gpu_cl_gcd(global, 32);
    gpu_cl_enqueue_nd(gr->gpu, gr->grid_hsl_id, 1, &local, &global, NULL);
    gpu_cl_read_buffer(gr->gpu, gr->width * gr->height * sizeof(*gr->rgba_cpu), 0, gr->rgba_cpu, gr->rgba_gpu);
    window_draw_from_bytes(gr->rgba_cpu, 0, 0, gr->width, gr->height);
}

void grid_renderer_bwr(grid_renderer *gr) {
    size_t global = gr->width * gr->height;
    size_t local = gpu_cl_gcd(global, 32);
    gpu_cl_enqueue_nd(gr->gpu, gr->grid_bwr_id, 1, &local, &global, NULL);
    gpu_cl_read_buffer(gr->gpu, gr->width * gr->height * sizeof(*gr->rgba_cpu), 0, gr->rgba_cpu, gr->rgba_gpu);
    window_draw_from_bytes(gr->rgba_cpu, 0, 0, gr->width, gr->height);
}

void grid_renderer_energy(grid_renderer *gr, double time) {
    size_t global = gr->g->gi.rows * gr->g->gi.cols;
    size_t local = gpu_cl_gcd(global, 32);

    gpu_cl_set_kernel_arg(gr->gpu, gr->calc_energy_id, 4, sizeof(time), &time);
    gpu_cl_enqueue_nd(gr->gpu, gr->calc_energy_id, 1, &local, &global, NULL);

    gpu_cl_read_buffer(gr->gpu, global * sizeof(*gr->buffer_cpu), 0, gr->buffer_cpu, gr->buffer_gpu);

    double min_energy = FLT_MAX;
    double max_energy = -FLT_MAX;

    for (uint64_t i = 0; i < global; ++i) {
        if (gr->buffer_cpu[i] < min_energy) min_energy = gr->buffer_cpu[i];
        if (gr->buffer_cpu[i] > max_energy) max_energy = gr->buffer_cpu[i];
    }

    global = gr->width * gr->height;
    local = gpu_cl_gcd(global, 32);
    
    gpu_cl_set_kernel_arg(gr->gpu, gr->energy_id, 3, sizeof(min_energy), &min_energy);
    gpu_cl_set_kernel_arg(gr->gpu, gr->energy_id, 4, sizeof(max_energy), &max_energy);
    gpu_cl_enqueue_nd(gr->gpu, gr->energy_id, 1, &local, &global, NULL);

    gpu_cl_read_buffer(gr->gpu, gr->width * gr->height * sizeof(*gr->rgba_cpu), 0, gr->rgba_cpu, gr->rgba_gpu);

    window_draw_from_bytes(gr->rgba_cpu, 0, 0, gr->width, gr->height);
}

void grid_renderer_charge(grid_renderer *gr) {
    size_t global = gr->g->gi.rows * gr->g->gi.cols;
    size_t local = gpu_cl_gcd(global, 32);

    gpu_cl_enqueue_nd(gr->gpu, gr->calc_charge_id, 1, &local, &global, NULL);

    gpu_cl_read_buffer(gr->gpu, global * sizeof(*gr->buffer_cpu), 0, gr->buffer_cpu, gr->buffer_gpu);

    double min_charge = FLT_MAX;
    double max_charge = -FLT_MAX;

    for (uint64_t i = 0; i < global; ++i) {
        if (gr->buffer_cpu[i] < min_charge) min_charge = gr->buffer_cpu[i];
        if (gr->buffer_cpu[i] > max_charge) max_charge = gr->buffer_cpu[i];
    }

    global = gr->width * gr->height;
    local = gpu_cl_gcd(global, 32);
    
    gpu_cl_set_kernel_arg(gr->gpu, gr->charge_id, 3, sizeof(min_charge), &min_charge);
    gpu_cl_set_kernel_arg(gr->gpu, gr->charge_id, 4, sizeof(max_charge), &max_charge);
    gpu_cl_enqueue_nd(gr->gpu, gr->charge_id, 1, &local, &global, NULL);

    gpu_cl_read_buffer(gr->gpu, gr->width * gr->height * sizeof(*gr->rgba_cpu), 0, gr->rgba_cpu, gr->rgba_gpu);

    window_draw_from_bytes(gr->rgba_cpu, 0, 0, gr->width, gr->height);

}

unsigned int steps_per_frame = 100;

void grid_renderer_gsa(grid *g, gsa_params params, unsigned int width, unsigned int height) {
    gpu_cl gpu_stack = gpu_cl_init(STR_NULL, params.field_func, STR_NULL, STR_NULL, params.compile_augment);
    gpu_cl *gpu = &gpu_stack;
    gsa_context ctx = gsa_context_init(g, gpu, params);
    window_init("GSA", width, height);

    grid_renderer gr = grid_renderer_init(g, gpu);
    int state = 'h';
    while (!window_should_close()) {
        switch (state) {
            case 'q':
                grid_renderer_charge(&gr);
                break;
            case 'e':
                grid_renderer_energy(&gr, 0.0);
                break;
            case 'h':
                grid_renderer_hsl(&gr);
                break;
            case 'b':
                grid_renderer_bwr(&gr);
                break;
            default:
                grid_renderer_hsl(&gr);
        }
        if (window_key_pressed('q'))
            state = 'q';
        else if (window_key_pressed('e'))
            state = 'e';
        else if (window_key_pressed('h'))
            state = 'h';
        else if (window_key_pressed('b'))
            state = 'b';

        for (unsigned int i = 0; i < steps_per_frame; ++i) {
            gsa_metropolis_step(&ctx);
            gsa_thermal_step(&ctx);
        }

        window_render();
        window_poll();
    }
    gsa_context_read_minimun_grid(&ctx);
    gsa_context_close(&ctx);
    grid_renderer_close(&gr);
    gpu_cl_close(gpu);
}

void grid_renderer_integrate(grid *g, integrate_params params, unsigned int width, unsigned int height) {
    gpu_cl gpu_stack = gpu_cl_init(params.current_func, params.field_func, params.temperature_func, STR_NULL, params.compile_augment);
    gpu_cl *gpu = &gpu_stack;
    window_init("Integration", width, height);
    integrate_context ctx = integrate_context_init(g, gpu, params);

    grid_renderer gr = grid_renderer_init(g, gpu);

    int state = 'b';
    while (!window_should_close()) {
        switch (state) {
            case 'q':
                grid_renderer_charge(&gr);
                break;
            case 'e':
                grid_renderer_energy(&gr, ctx.time);
                break;
            case 'h':
                grid_renderer_hsl(&gr);
                break;
            case 'b':
                grid_renderer_bwr(&gr);
                break;
            default:
                grid_renderer_hsl(&gr);
        }
        if (window_key_pressed('q'))
            state = 'q';
        else if (window_key_pressed('e'))
            state = 'e';
        else if (window_key_pressed('h'))
            state = 'h';
        else if (window_key_pressed('b'))
            state = 'b';

        for (unsigned int i = 0; i < steps_per_frame; ++i) {
            integrate_step(&ctx);
            integrate_exchange_grids(&ctx);
            ctx.time += params.dt;
        }

        window_render();
        window_poll();
    }
    integrate_context_close(&ctx);
    grid_renderer_close(&gr);
    gpu_cl_close(gpu);
}

void grid_renderer_gradient_descent(grid *g, gradient_descent_params params, unsigned int width, unsigned int height) {
    gpu_cl gpu_stack = gpu_cl_init(STR_NULL, params.field_func, STR_NULL, STR_NULL, params.compile_augment);
    gpu_cl *gpu = &gpu_stack;
    window_init("Gradient Descent", width, height);
    gradient_descent_context ctx = gradient_descent_context_init(g, gpu, params);
    grid_renderer gr = grid_renderer_init(g, gpu);
    int state = 'h';
    while (!window_should_close()) {
        switch (state) {
            case 'q':
                grid_renderer_charge(&gr);
                break;
            case 'e':
                grid_renderer_energy(&gr, 0.0);
                break;
            case 'h':
                grid_renderer_hsl(&gr);
                break;
            case 'b':
                grid_renderer_bwr(&gr);
                break;
            default:
                grid_renderer_hsl(&gr);
        }
        if (window_key_pressed('q'))
            state = 'q';
        else if (window_key_pressed('e'))
            state = 'e';
        else if (window_key_pressed('h'))
            state = 'h';
        else if (window_key_pressed('b'))
            state = 'b';

        for (unsigned int i = 0; i < steps_per_frame; ++i) {
            gradient_descent_step(&ctx);
            gradient_descent_exchange(&ctx);
            if (ctx.step % 1000 == 0)
                logging_log(LOG_INFO, "%"PRIu64" Gradient Descent %"PRIu64" - Min Energy: %e eV - Temperature %e", ctx.outer_step, ctx.step, ctx.min_energy, ctx.params.T);
        }

        window_render();
        window_poll();
    }
    gradient_descent_read_mininum_grid(&ctx);
    gradient_descent_close(&ctx);
    grid_renderer_close(&gr);
    gpu_cl_close(gpu);
}
