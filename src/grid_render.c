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
    grid_to_gpu(g, *ret.gpu);

    ret.r_global = ret.width * ret.height;
    ret.r_global = ret.r_global + (gpu_optimal_wg - ret.r_global % gpu_optimal_wg);
    
    ret.g_global = g->gi.rows * g->gi.cols;
    ret.g_global = ret.g_global + (gpu_optimal_wg - ret.g_global % gpu_optimal_wg);

    ret.local = gpu_optimal_wg;

    ret.rgba_cpu = mmalloc(ret.width * ret.height * sizeof(*ret.rgba_cpu));
    ret.rgba_gpu = gpu_cl_create_gpu(ret.gpu, sizeof(*ret.rgba_cpu) * ret.width * ret.height, CL_MEM_READ_WRITE);

    ret.buffer_cpu = mmalloc(g->gi.rows * g->gi.cols * sizeof(*ret.buffer_cpu));
    ret.buffer_gpu = gpu_cl_create_gpu(ret.gpu, sizeof(*ret.buffer_cpu) * g->gi.rows * g->gi.cols, CL_MEM_READ_WRITE);

    ret.v3d_buffer_cpu = mmalloc(g->gi.rows * g->gi.cols * sizeof(*ret.v3d_buffer_cpu));
    ret.v3d_buffer_gpu = gpu_cl_create_gpu(ret.gpu, sizeof(*ret.v3d_buffer_cpu) * g->gi.rows * g->gi.cols, CL_MEM_READ_WRITE);

    ret.grid_hsl_id = gpu_cl_append_kernel(ret.gpu, "render_grid_hsl");
    ret.grid_bwr_id = gpu_cl_append_kernel(ret.gpu, "render_grid_bwr");
    ret.pinning_id = gpu_cl_append_kernel(ret.gpu, "render_pinning");
    ret.energy_id = gpu_cl_append_kernel(ret.gpu, "render_energy");
    ret.charge_id = gpu_cl_append_kernel(ret.gpu, "render_charge");
    ret.electric_id = gpu_cl_append_kernel(ret.gpu, "render_electric");
    ret.calc_charge_id = gpu_cl_append_kernel(ret.gpu, "calculate_charge_to_render");
    ret.calc_energy_id = gpu_cl_append_kernel(ret.gpu, "calculate_energy");
    ret.calc_electric_id = gpu_cl_append_kernel(ret.gpu, "calculate_electric");

    gpu_cl_fill_kernel_args(ret.gpu, ret.grid_hsl_id, 0, 5, &ret.g->m_gpu, sizeof(cl_mem), &ret.g->gi, sizeof(ret.g->gi), &ret.rgba_gpu, sizeof(cl_mem), &ret.width, sizeof(ret.width), &ret.height, sizeof(ret.height));

    gpu_cl_fill_kernel_args(ret.gpu, ret.grid_bwr_id, 0, 5, &ret.g->m_gpu, sizeof(cl_mem), &ret.g->gi, sizeof(ret.g->gi), &ret.rgba_gpu, sizeof(cl_mem), &ret.width, sizeof(ret.width), &ret.height, sizeof(ret.height));

    gpu_cl_fill_kernel_args(ret.gpu, ret.pinning_id, 0, 6, &ret.g->gp_gpu, sizeof(cl_mem), &ret.g->gi.rows, sizeof(ret.g->gi.rows), &ret.g->gi.cols, sizeof(ret.g->gi.cols), &ret.rgba_gpu, sizeof(cl_mem), &ret.width, sizeof(ret.width), &ret.height, sizeof(ret.height));

    gpu_cl_fill_kernel_args(ret.gpu, ret.calc_charge_id, 0, 3, &ret.g->m_gpu, sizeof(cl_mem), &ret.g->gi, sizeof(ret.g->gi), &ret.buffer_gpu, sizeof(cl_mem));

    //Need to set time
    gpu_cl_fill_kernel_args(ret.gpu, ret.calc_energy_id, 0, 4, &ret.g->gp_gpu, sizeof(cl_mem), &ret.g->m_gpu, sizeof(cl_mem), &ret.g->gi, sizeof(ret.g->gi), &ret.buffer_gpu, sizeof(cl_mem));


    gpu_cl_fill_kernel_args(ret.gpu, ret.charge_id, 0, 3, &ret.buffer_gpu, sizeof(cl_mem), &ret.g->gi.rows, sizeof(ret.g->gi.rows), &ret.g->gi.cols, sizeof(ret.g->gi.cols));
    gpu_cl_fill_kernel_args(ret.gpu, ret.charge_id, 5, 3, &ret.rgba_gpu, sizeof(cl_mem), &ret.width, sizeof(ret.width), &ret.height, sizeof(ret.height)); //Need to set ranges

    gpu_cl_fill_kernel_args(ret.gpu, ret.energy_id, 0, 3, &ret.buffer_gpu, sizeof(cl_mem), &ret.g->gi.rows, sizeof(ret.g->gi.rows), &ret.g->gi.cols, sizeof(ret.g->gi.cols));
    gpu_cl_fill_kernel_args(ret.gpu, ret.energy_id, 5, 3, &ret.rgba_gpu, sizeof(cl_mem), &ret.width, sizeof(ret.width), &ret.height, sizeof(ret.height)); //Need to set ranges
                                                                                                                                                          //
//kernel void calculate_electric(GLOBAL grid_site_params *gs, GLOBAL v3d *m0, GLOBAL v3d *m1, GLOBAL v3d *out, double dt, grid_info gi) {
    gpu_cl_fill_kernel_args(ret.gpu, ret.calc_electric_id, 0, 2, &ret.g->gp_gpu, sizeof(cl_mem), &ret.g->m_gpu, sizeof(cl_mem));
    gpu_cl_set_kernel_arg(ret.gpu, ret.calc_electric_id, 3, sizeof(cl_mem), &ret.v3d_buffer_gpu);
    gpu_cl_set_kernel_arg(ret.gpu, ret.calc_electric_id, 5, sizeof(ret.g->gi), &ret.g->gi);

//    kernel void render_electric(GLOBAL v3d *field, unsigned int rows, unsigned int cols, double max_mod,
//                            GLOBAL RGBA32 *rgba, unsigned int width, unsigned int height) {

    gpu_cl_fill_kernel_args(ret.gpu, ret.electric_id, 0, 3, &ret.v3d_buffer_gpu, sizeof(cl_mem), &ret.g->gi.rows, sizeof(ret.g->gi.rows), &ret.g->gi.cols, sizeof(ret.g->gi.cols));
    gpu_cl_fill_kernel_args(ret.gpu, ret.electric_id, 4, 3, &ret.rgba_gpu, sizeof(cl_mem), &ret.width, sizeof(ret.width), &ret.height, sizeof(ret.height));

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
    gpu_cl_enqueue_nd(gr->gpu, gr->grid_hsl_id, 1, &gr->local, &gr->r_global, NULL);
    gpu_cl_read_gpu(gr->gpu, gr->width * gr->height * sizeof(*gr->rgba_cpu), 0, gr->rgba_cpu, gr->rgba_gpu);
    window_draw_from_bytes(gr->rgba_cpu, 0, 0, gr->width, gr->height);
}

void grid_renderer_pinning(grid_renderer *gr) {
    gpu_cl_enqueue_nd(gr->gpu, gr->pinning_id, 1, &gr->local, &gr->r_global, NULL);

    gpu_cl_read_gpu(gr->gpu, gr->width * gr->height * sizeof(*gr->rgba_cpu), 0, gr->rgba_cpu, gr->rgba_gpu);
    window_draw_from_bytes(gr->rgba_cpu, 0, 0, gr->width, gr->height);
}

void grid_renderer_bwr(grid_renderer *gr) {
    gpu_cl_enqueue_nd(gr->gpu, gr->grid_bwr_id, 1, &gr->local, &gr->r_global, NULL);
    gpu_cl_read_gpu(gr->gpu, gr->width * gr->height * sizeof(*gr->rgba_cpu), 0, gr->rgba_cpu, gr->rgba_gpu);
    window_draw_from_bytes(gr->rgba_cpu, 0, 0, gr->width, gr->height);
}

void grid_renderer_energy(grid_renderer *gr, double time) {
    gpu_cl_set_kernel_arg(gr->gpu, gr->calc_energy_id, 4, sizeof(time), &time);
    gpu_cl_enqueue_nd(gr->gpu, gr->calc_energy_id, 1, &gr->local, &gr->g_global, NULL);

    gpu_cl_read_gpu(gr->gpu, gr->g->gi.rows * gr->g->gi.cols * sizeof(*gr->buffer_cpu), 0, gr->buffer_cpu, gr->buffer_gpu);

    double min_energy = FLT_MAX;
    double max_energy = -FLT_MAX;

    for (uint64_t i = 0; i < gr->g->gi.rows * gr->g->gi.cols; ++i) {
        if (gr->buffer_cpu[i] < min_energy) min_energy = gr->buffer_cpu[i];
        if (gr->buffer_cpu[i] > max_energy) max_energy = gr->buffer_cpu[i];
    }

    gpu_cl_set_kernel_arg(gr->gpu, gr->energy_id, 3, sizeof(min_energy), &min_energy);
    gpu_cl_set_kernel_arg(gr->gpu, gr->energy_id, 4, sizeof(max_energy), &max_energy);
    gpu_cl_enqueue_nd(gr->gpu, gr->energy_id, 1, &gr->local, &gr->r_global, NULL);

    gpu_cl_read_gpu(gr->gpu, gr->width * gr->height * sizeof(*gr->rgba_cpu), 0, gr->rgba_cpu, gr->rgba_gpu);

    window_draw_from_bytes(gr->rgba_cpu, 0, 0, gr->width, gr->height);
}

void grid_renderer_charge(grid_renderer *gr) {
    gpu_cl_enqueue_nd(gr->gpu, gr->calc_charge_id, 1, &gr->local, &gr->g_global, NULL);

    gpu_cl_read_gpu(gr->gpu, gr->g->gi.rows * gr->g->gi.cols * sizeof(*gr->buffer_cpu), 0, gr->buffer_cpu, gr->buffer_gpu);

    double min_charge = FLT_MAX;
    double max_charge = -FLT_MAX;

    for (uint64_t i = 0; i < gr->g->gi.rows * gr->g->gi.cols; ++i) {
        if (gr->buffer_cpu[i] < min_charge) min_charge = gr->buffer_cpu[i];
        if (gr->buffer_cpu[i] > max_charge) max_charge = gr->buffer_cpu[i];
    }

    gpu_cl_set_kernel_arg(gr->gpu, gr->charge_id, 3, sizeof(min_charge), &min_charge);
    gpu_cl_set_kernel_arg(gr->gpu, gr->charge_id, 4, sizeof(max_charge), &max_charge);
    gpu_cl_enqueue_nd(gr->gpu, gr->charge_id, 1, &gr->local, &gr->r_global, NULL);

    gpu_cl_read_gpu(gr->gpu, gr->width * gr->height * sizeof(*gr->rgba_cpu), 0, gr->rgba_cpu, gr->rgba_gpu);

    window_draw_from_bytes(gr->rgba_cpu, 0, 0, gr->width, gr->height);

}

void grid_renderer_electric_field(grid_renderer *gr) {
    gpu_cl_enqueue_nd(gr->gpu, gr->calc_electric_id, 1, &gr->local, &gr->g_global, NULL);

    gpu_cl_read_gpu(gr->gpu, gr->g->gi.rows * gr->g->gi.cols * sizeof(*gr->v3d_buffer_cpu), 0, gr->v3d_buffer_cpu, gr->v3d_buffer_gpu);

    double max_electric = -FLT_MAX;
   // for (uint64_t i = 0; i < global; ++i) {
   //     double value = v3d_dot(gr->v3d_buffer_cpu[i], gr->v3d_buffer_cpu[i]);
   //     v3d v = gr->v3d_buffer_cpu[i];
   //     if (value > max_electric) max_electric = value;
   // }

    gpu_cl_set_kernel_arg(gr->gpu, gr->electric_id, 3, sizeof(max_electric), &max_electric);
    gpu_cl_enqueue_nd(gr->gpu, gr->electric_id, 1, &gr->local, &gr->r_global, NULL);

    gpu_cl_read_gpu(gr->gpu, gr->width * gr->height * sizeof(*gr->rgba_cpu), 0, gr->rgba_cpu, gr->rgba_gpu);

    window_draw_from_bytes(gr->rgba_cpu, 0, 0, gr->width, gr->height);
}

double eps = 0.105;

void grid_renderer_clustering(grid_renderer *gr) {
    double delta_h = 1.0 / gr->g->clusters.len;
    for (uint64_t y = 0; y < gr->height; ++y) {
        uint64_t gy = (gr->height - y - 1) / (double)gr->height * gr->g->gi.rows;
        for (uint64_t x = 0; x < gr->width; ++x) {
            uint64_t gx = x / (double)gr->width * gr->g->gi.cols;
            uint64_t c_idx = gr->g->points[gy * gr->g->gi.cols + gx].cluster;
            if (gr->g->points[gy * gr->g->gi.cols + gx].label != NOISE)
                gr->rgba_cpu[y * gr->width + x] = m_to_hsl(v3d_normalize(gr->g->clusters.items[c_idx].avg_m));
            else
                gr->rgba_cpu[y * gr->width + x] = (RGBA32){.r = 255, .g = 0, .b = 0, .a = 255};
        }
    }

    for (uint64_t i = 0; i < gr->g->clusters.len; ++i) {
        for (int idy = -10; idy <= 10; ++idy) {
            int iy = (gr->g->clusters.items[i].y + gr->g->gp->lattice * 0.5) / (gr->g->gi.rows) * (gr->height - 1) * 1.0 / gr->g->gp->lattice;
            iy = iy + idy;
            iy = gr->height - iy - 1;
            for (int idx = -10; idx <= 10; ++idx) {
                int ix = (gr->g->clusters.items[i].x + gr->g->gp->lattice * 0.5) / (gr->g->gi.cols) * (gr->width - 1) * 1.0 / gr->g->gp->lattice;
                ix = ix + idx;
                RGBA32 color = {0};

                if (idx * idx + idy * idy <= 8 * 8)
                    color = m_to_hsl(v3d_normalize(gr->g->clusters.items[i].avg_m));
                else if (idx * idx + idy * idy <= 10 * 10)
                    color = (RGBA32){.bgra = ~m_to_hsl(v3d_normalize(gr->g->clusters.items[i].avg_m)).bgra};
                else
                    continue;
                if (iy * gr->width + ix < gr->width * gr->height)
                    gr->rgba_cpu[iy * gr->width + ix] = color;

            }
        }
    }

    window_draw_from_bytes(gr->rgba_cpu, 0, 0, gr->width, gr->height);
}

void grid_renderer_clustering_centers(grid_renderer *gr) {
    for (uint64_t i = 0; i < gr->g->clusters.len; ++i) {
        for (int idy = -10; idy <= 10; ++idy) {
            int iy = (gr->g->clusters.items[i].y + gr->g->gp->lattice * 0.5) / (gr->g->gi.rows) * (gr->height - 1) * 1.0 / gr->g->gp->lattice;
            iy = iy + idy;
            iy = gr->height - iy - 1;
            for (int idx = -10; idx <= 10; ++idx) {
                int ix = (gr->g->clusters.items[i].x + gr->g->gp->lattice * 0.5) / (gr->g->gi.cols) * (gr->width - 1) * 1.0 / gr->g->gp->lattice;
                ix = ix + idx;
                RGBA32 color = {0};

                if (idx * idx + idy * idy <= 8 * 8)
                    color = m_to_hsl(v3d_normalize(gr->g->clusters.items[i].avg_m));
                else if (idx * idx + idy * idy <= 10 * 10)
                    color = (RGBA32){.bgra = ~m_to_hsl(v3d_normalize(gr->g->clusters.items[i].avg_m)).bgra};
                else
                    continue;
                if (iy * gr->width + ix < gr->width * gr->height)
                    gr->rgba_cpu[iy * gr->width + ix] = color;

            }
        }
    }
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
//kernel void calculate_electric(GLOBAL grid_site_params *gs, GLOBAL v3d *m0, GLOBAL v3d *m1, GLOBAL v3d *out, double dt, grid_info gi) {
    gpu_cl_set_kernel_arg(gpu, gr.calc_electric_id, 2, sizeof(cl_mem), &ctx.swap_gpu);
    gpu_cl_set_kernel_arg(gpu, gr.calc_electric_id, 4, sizeof(ctx.params.dt), &ctx.params.dt);

    int state = 'h';
    integrate_step(&ctx);
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
            case 'w':
                grid_renderer_electric_field(&gr);
                break;
            case 'c':
                grid_renderer_clustering(&gr);
                break;
            case 'v':
                grid_renderer_hsl(&gr);
                grid_renderer_clustering_centers(&gr);
                break;
            default:
                grid_renderer_hsl(&gr);
                break;
        }
        //grid_renderer_pinning(&gr);
        if (window_key_pressed('q'))
            state = 'q';
        else if (window_key_pressed('e'))
            state = 'e';
        else if (window_key_pressed('h'))
            state = 'h';
        else if (window_key_pressed('b'))
            state = 'b';
        else if (window_key_pressed('w'))
            state = 'w';
        else if (window_key_pressed('c'))
            state = 'c';
        else if (window_key_pressed('v'))
            state = 'v';

        if (window_key_pressed('k')) {
            eps += 0.01;
            logging_log(LOG_INFO, "eps: %e" , eps);
        } else if (window_key_pressed('l')) {
            eps -= 0.01;
            logging_log(LOG_INFO, "eps: %e" , eps);
        } 

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
