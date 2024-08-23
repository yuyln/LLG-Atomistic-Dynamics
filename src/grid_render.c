#include "grid_render.h"
#include "colors.h"
#include "complete_kernel.h"
#include "constants.h"
#include "kernel_funcs.h"
#include "gsa.h"
#include "integrate.h"
#include "gradient_descent.h"
#include "profiler.h"
#include "allocator.h"

#include <float.h>
#include <inttypes.h>
#include <stdint.h>

grid_renderer grid_renderer_init(grid *g, gpu_cl *gpu) {
    grid_renderer ret = {0};
    ret.width = window_width();
    ret.height = window_height();
    ret.g = g;
    ret.gpu = gpu;
    ret.k = 0;
    grid_to_gpu(g, *ret.gpu);

    ret.r_global = ret.width * ret.height;
    ret.r_global = ret.r_global + (gpu_optimal_wg - ret.r_global % gpu_optimal_wg);
    
    ret.g_global = g->dimensions;
    ret.g_global = ret.g_global + (gpu_optimal_wg - ret.g_global % gpu_optimal_wg);

    ret.local = gpu_optimal_wg;

    ret.rgba_cpu = mmalloc(ret.width * ret.height * sizeof(*ret.rgba_cpu));
    ret.rgba_gpu = gpu_cl_create_gpu(ret.gpu, sizeof(*ret.rgba_cpu) * ret.width * ret.height, CL_MEM_READ_WRITE);

    ret.buffer_cpu = mmalloc(g->dimensions * sizeof(*ret.buffer_cpu));
    ret.buffer_gpu = gpu_cl_create_gpu(ret.gpu, sizeof(*ret.buffer_cpu) * g->dimensions, CL_MEM_READ_WRITE);

    ret.v3d_buffer_cpu = mmalloc(g->dimensions * sizeof(*ret.v3d_buffer_cpu));
    ret.v3d_buffer_gpu = gpu_cl_create_gpu(ret.gpu, sizeof(*ret.v3d_buffer_cpu) * g->dimensions, CL_MEM_READ_WRITE);

    ret.grid_hsl_id = gpu_cl_append_kernel(ret.gpu, "render_grid_hsl");
    ret.grid_bwr_id = gpu_cl_append_kernel(ret.gpu, "render_grid_bwr");

    gpu_cl_fill_kernel_args(ret.gpu, ret.grid_hsl_id, 0, 6, &ret.g->m_gpu, sizeof(cl_mem), &ret.g->gi, sizeof(ret.g->gi), &ret.k, sizeof(ret.k), &ret.rgba_gpu, sizeof(cl_mem), &ret.width, sizeof(ret.width), &ret.height, sizeof(ret.height));

    gpu_cl_fill_kernel_args(ret.gpu, ret.grid_bwr_id, 0, 6, &ret.g->m_gpu, sizeof(cl_mem), &ret.g->gi, sizeof(ret.g->gi), &ret.k, sizeof(ret.k), &ret.rgba_gpu, sizeof(cl_mem), &ret.width, sizeof(ret.width), &ret.height, sizeof(ret.height));

    return ret;
}

void grid_renderer_close(grid_renderer *gr) {
    mfree(gr->rgba_cpu);
    mfree(gr->buffer_cpu);
    mfree(gr->v3d_buffer_cpu);
    gpu_cl_release_memory(gr->rgba_gpu);
    gpu_cl_release_memory(gr->buffer_gpu);
    gpu_cl_release_memory(gr->v3d_buffer_gpu);
    grid_release_from_gpu(gr->g);
}

void grid_renderer_hsl(grid_renderer *gr) {
    gpu_cl_set_kernel_arg(gr->gpu, gr->grid_hsl_id, 2, sizeof(gr->k), &gr->k);
    gpu_cl_enqueue_nd(gr->gpu, gr->grid_hsl_id, 1, &gr->local, &gr->r_global, NULL);
    gpu_cl_read_gpu(gr->gpu, gr->width * gr->height * sizeof(*gr->rgba_cpu), 0, gr->rgba_cpu, gr->rgba_gpu);
    window_draw_from_bytes(gr->rgba_cpu, 0, 0, gr->width, gr->height);
}

void grid_renderer_bwr(grid_renderer *gr) {
    gpu_cl_set_kernel_arg(gr->gpu, gr->grid_bwr_id, 2, sizeof(gr->k), &gr->k);
    gpu_cl_enqueue_nd(gr->gpu, gr->grid_bwr_id, 1, &gr->local, &gr->r_global, NULL);
    gpu_cl_read_gpu(gr->gpu, gr->width * gr->height * sizeof(*gr->rgba_cpu), 0, gr->rgba_cpu, gr->rgba_gpu);
    window_draw_from_bytes(gr->rgba_cpu, 0, 0, gr->width, gr->height);
}

unsigned int steps_per_frame = 100;
double print_time = 1.0;

void grid_renderer_gsa(grid *g, gsa_params params, unsigned int width, unsigned int height) {
    gpu_cl gpu_stack = gpu_cl_init(NULL, params.field_func, NULL, NULL, params.compile_augment);
    gpu_cl *gpu = &gpu_stack;
    gsa_context ctx = gsa_context_init(g, gpu, params);
    window_init("GSA", width, height);

    grid_renderer gr = grid_renderer_init(g, gpu);
    int state = 'h';

    double print_timer = 0;
    double dt_fps = 0;
    double frame_start = profiler_get_sec();
    uint64_t frames = 0;

    while (!window_should_close()) {
        switch (state) {
            case 'h':
                grid_renderer_hsl(&gr);
                break;
            case 'b':
                grid_renderer_bwr(&gr);
                break;
            default:
                grid_renderer_hsl(&gr);
        }
        if (window_key_pressed('h'))
            state = 'h';
        else if (window_key_pressed('b'))
            state = 'b';

        if (window_key_pressed('k'))
            gr.k = (gr.k + 1) % g->gi.depth;

        for (unsigned int i = 0; i < steps_per_frame; ++i) {
            gsa_metropolis_step(&ctx);
            gsa_thermal_step(&ctx);
        }

        if (print_timer >= print_time) {
            logging_log(LOG_INFO, "GSA FPS: %"PRIu64" <dt real>: %e", frames / print_timer, print_timer / frames);
            print_timer = 0;
            frames = 0;
        }

        window_render();
        window_poll();
        frames++;
        double end = profiler_get_sec();
        dt_fps = end - frame_start;
        print_timer += dt_fps;
        frame_start = end;
    }
    gsa_context_read_minimun_grid(&ctx);
    gsa_context_close(&ctx);
    grid_renderer_close(&gr);
    gpu_cl_close(gpu);
}

void grid_renderer_integrate(grid *g, integrate_params params, unsigned int width, unsigned int height) {
    gpu_cl gpu_stack = gpu_cl_init(params.current_func, params.field_func, params.temperature_func, NULL, params.compile_augment);
    gpu_cl *gpu = &gpu_stack;
    window_init("Integration", width, height);
    integrate_context ctx = integrate_context_init(g, gpu, params);

    grid_renderer gr = grid_renderer_init(g, gpu);
    
    double print_timer = 0;
    double dt_fps = 0;
    double frame_start = profiler_get_sec();
    uint64_t frames = 0;

    int state = 'h';
    integrate_step(&ctx);
    while (!window_should_close()) {
        switch (state) {
            case 'h':
                grid_renderer_hsl(&gr);
                break;
            case 'b':
                grid_renderer_bwr(&gr);
                break;
            default:
                grid_renderer_hsl(&gr);
        }
        if (window_key_pressed('h'))
            state = 'h';
        else if (window_key_pressed('b'))
            state = 'b';

        if (window_key_pressed('k'))
            gr.k = (gr.k + 1) % g->gi.depth;

        for (unsigned int i = 0; i < steps_per_frame; ++i) {
            integrate_exchange_grids(&ctx);
            integrate_step(&ctx);
        }

        if (print_timer >= print_time) {
            logging_log(LOG_INFO, "Integrate FPS: %"PRIu64, (uint64_t)(frames / print_timer));
            logging_log(LOG_INFO, "Integrate Steps per Second: %"PRIu64, (uint64_t)(frames / print_timer * steps_per_frame));
            logging_log(LOG_INFO, "Integrate <dt_real>: %es", print_timer / frames);
            logging_log(LOG_INFO, "Integrate time: %ens", ctx.time / NS);
            print_timer = 0;
            frames = 0;
        }

        window_render();
        window_poll();

        frames++;
        double end = profiler_get_sec();
        dt_fps = end - frame_start;
        print_timer += dt_fps;
        frame_start = end;
    }
    integrate_context_close(&ctx);
    grid_renderer_close(&gr);
}

void grid_renderer_gradient_descent(grid *g, gradient_descent_params params, unsigned int width, unsigned int height) {
    gpu_cl gpu_stack = gpu_cl_init(NULL, params.field_func, NULL, NULL, params.compile_augment);
    gpu_cl *gpu = &gpu_stack;
    window_init("Gradient Descent", width, height);
    gradient_descent_context ctx = gradient_descent_context_init(g, gpu, params);
    grid_renderer gr = grid_renderer_init(g, gpu);

    double print_timer = 0;
    double dt_fps = 0;
    double frame_start = profiler_get_sec();
    uint64_t frames = 0;

    int state = 'h';
    while (!window_should_close()) {
        switch (state) {
            case 'h':
                grid_renderer_hsl(&gr);
                break;
            case 'b':
                grid_renderer_bwr(&gr);
                break;
            default:
                grid_renderer_hsl(&gr);
        }
        if (window_key_pressed('h'))
            state = 'h';
        else if (window_key_pressed('b'))
            state = 'b';
        
        if (window_key_pressed('k'))
            gr.k = (gr.k + 1) % g->gi.depth;

        for (unsigned int i = 0; i < steps_per_frame; ++i) {
            gradient_descent_step(&ctx);
            gradient_descent_exchange(&ctx);
        }
        if (print_timer >= print_time) {
            logging_log(LOG_INFO, "Gradient Descent FPS: %"PRIu64, (uint64_t)(frames / print_timer));
            logging_log(LOG_INFO, "Gradient Descent Steps per Second: %"PRIu64, (uint64_t)(frames / print_timer * steps_per_frame));
            logging_log(LOG_INFO, "Gradient Descent <dt_real>: %es", print_timer / frames);
            logging_log(LOG_INFO, "Gradient Descent[Outer=%"PRIu64"][Inner=%"PRIu64"] Minimun Energy: %e eV Temperature: %e",
                                   ctx.outer_step, ctx.step, ctx.min_energy / QE, ctx.params.T);
            print_timer = 0;
            frames = 0;
        }

        window_render();
        window_poll();

        frames++;
        double end = profiler_get_sec();
        dt_fps = end - frame_start;
        print_timer += dt_fps;
        frame_start = end;
    }
    gradient_descent_read_mininum_grid(&ctx);
    gradient_descent_close(&ctx);
    grid_renderer_close(&gr);
    gpu_cl_close(gpu);
}
