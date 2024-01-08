#define __PROFILER_IMPLEMENTATION
#include "atomistic_simulation.h"
#include <sys/time.h>
#include <time.h>

void run_gsa(grid *g, gpu_cl *gpu) {
    double ratio = (double)g->gi.cols / g->gi.rows;
    render_window *window = window_init(800 * ratio, 800);

    grid_renderer gr = grid_renderer_init(g, gpu, window);
    gsa_context ctx = gsa_context_init(g, gr.gpu, .T0 = 500.0, .inner_steps=700000, .qV = 2.7, .print_factor=10);

    struct timespec current_time;
    clock_gettime(CLOCK_REALTIME, &current_time);
    int state = 'h';

    double time_for_print = 1.0;
    double stopwatch_print = -1.0;
    int frames = 0;
    const int steps = 100;

    while(!window_should_close(window)) {
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
        if (window_key_pressed(window, 'q'))
            state = 'q';
        else if (window_key_pressed(window, 'e'))
            state = 'e';
        else if (window_key_pressed(window, 'h'))
            state = 'h';
        else if (window_key_pressed(window, 'b'))
            state = 'b';

        for (int i = 0; i < steps; ++i) {
            gsa_metropolis_step(&ctx);
            gsa_thermal_step(&ctx);
        }

        window_render(window);
        window_poll(window);

        struct timespec new_time;
        clock_gettime(CLOCK_REALTIME, &new_time);
        double dt_real = (new_time.tv_sec + new_time.tv_nsec * 1.0e-9) - (current_time.tv_sec + current_time.tv_nsec * 1.0e-9);
        current_time = new_time;
        stopwatch_print += dt_real;
        frames++;
        if (stopwatch_print >= 0) {
            printf("FPS: %d - Frame Time: %e ms\n", (int)(frames / time_for_print), time_for_print / frames / 1.0e-3);
            stopwatch_print = -1.0;
            frames = 0;
        }
    }

    gsa_context_read_minimun_grid(&ctx);
    grid_release_from_gpu(g);
    gsa_context_clear(&ctx);
    grid_renderer_close(&gr);
    window_close(window);
}

void run_integration(grid *g, gpu_cl *gpu, double dt) {
    double ratio = (double)g->gi.cols / g->gi.rows;
    render_window *window = window_init(800 * ratio, 800);

    grid_renderer gr = grid_renderer_init(g, gpu, window);
    integrate_context ctx = integrate_context_init(g, gr.gpu, dt);

    struct timespec current_time;
    clock_gettime(CLOCK_REALTIME, &current_time);
    int state = 'h';

    double time_for_print = 1.0;
    double stopwatch_print = -1.0;
    int frames = 0;
    const int steps = 100;

    while(!window_should_close(window)) {
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
        if (window_key_pressed(window, 'q'))
            state = 'q';
        else if (window_key_pressed(window, 'e'))
            state = 'e';
        else if (window_key_pressed(window, 'h'))
            state = 'h';
        else if (window_key_pressed(window, 'b'))
            state = 'b';

        for (int i = 0; i < steps; ++i) {
            integrate_step(&ctx);
            integrate_exchange_grids(&ctx);
            ctx.time += dt;
        }

        window_render(window);
        window_poll(window);

        struct timespec new_time;
        clock_gettime(CLOCK_REALTIME, &new_time);
        double dt_real = (new_time.tv_sec + new_time.tv_nsec * 1.0e-9) - (current_time.tv_sec + current_time.tv_nsec * 1.0e-9);
        current_time = new_time;
        stopwatch_print += dt_real;
        frames++;
        if (stopwatch_print >= 0) {
            printf("FPS: %d - Frame Time: %e ms - System Steps: %d - System dt: %e - System Time: %e\n", (int)(frames / time_for_print), time_for_print / frames / 1.0e-3, steps * frames, frames * steps * dt, ctx.time);
            stopwatch_print = -1.0;
            frames = 0;
        }
    }
    integrate_context_read_grid(&ctx);
    grid_release_from_gpu(g);
    integrate_context_close(&ctx);
    grid_renderer_close(&gr);
    window_close(window);
}

//@TODO: Change openclwrapper to print file and location correctly
//@TODO: Check uint64_t->int changes
//@TODO: Do 3D
//@TODO: Clear everything on integrate context and gsa context(done?)
int main(void) {
    int rows = 32;
    int cols = 32;
    double dt = HBAR / (1.0e-3 * QE) * 0.01;

    grid g = grid_init(rows, cols);

    grid_set_dm(&g, 1.0 * QE * 1.0e-3, 0.0, R_ij_CROSS_Z);
    grid_set_anisotropy(&g, (anisotropy){.ani = 0.02 * QE * 1.0e-3, .dir = v3d_c(0.0, 0.0, 1.0)});
    v3d_fill_with_random(g.m, rows, cols);

    string_view current_func = sv_from_cstr("current ret = (current){0};\n"\
                                            "return ret;\n"\
                                             "ret.type = CUR_STT;\n"\
                                             "time -= 0.1 * NS;\n"\
                                             "double j_ac = 5.0e10 * (time > 0);\n"\
                                             "double j_dc = 1.0e10 * (time > 0);\n"\
                                             "double omega = 319145920.365;\n"\
                                             "ret.stt.j = v3d_c(j_dc, j_ac * sin(omega * time), 0.0);\n"\
                                             "ret.stt.polarization = -1.0;\n"\
                                             "ret.stt.beta = 0.0;\n"\
                                             "return ret;");
    string_view field_func = sv_from_cstr("double normalized = 0.5;\n"\
                                          "double real = normalized * gs.dm * gs.dm / gs.exchange / gs.mu;\n"\
                                          "//double osc = sin(5 * M_PI * gs.col / 64.0 - M_PI * 1.0 * time / NS);\n"\
                                          "//real = real * (1.0 + 0.1 * osc);\n"\
                                          "return v3d_c(0.0, 0.0, real);");
    string_view compile = sv_from_cstr("-cl-fast-relaxed-math");

    //integrate(&g, .dt = dt, .duration = 1 * NS, .current_generation_function = current_func, .field_generation_function = field_func, .compile_augment = compile);

    srand(time(NULL));

    gpu_cl gpu = gpu_cl_init(current_func, field_func, sv_from_cstr(""), compile);
    //run_gsa(&g, &gpu);
    run_integration(&g, &gpu, dt);

    grid_free(&g);
    return 0;
}
