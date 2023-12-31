#define __PROFILER_IMPLEMENTATION
#include "atomistic_simulation.h"
#include <sys/time.h>
#include <time.h>

//@TODO: Change openclwrapper to print file and location correctly
//@TODO: Check uint64_t->int changes
//@TODO: Do 3D
int main(void) {
    double dt = HBAR / (1.0e-3 * QE) * 0.01;
    int rows = 272;
    int cols = 272;
    double ratio = (double)cols / rows;
    render_window *window = window_init(800, 800 / ratio);
    grid g = grid_init(rows, cols);
    g.gi.pbc.m = v3d_normalize(v3d_c(1.0, 0.0, 0.0));
    for (int i = 0; i < rows * cols; ++i)
        g.m[i] = v3d_c(0.0, 0.0, 1.0);
    v3d_create_skyrmion(g.m, g.gi.rows, g.gi.cols, 10, rows / 2.0, cols / 4.0, -1.0, 1.0, M_PI / 2.0);

    grid_set_dm(&g, 0.18 * QE * 1.0e-3, 0.0, R_ij_CROSS_Z);
    grid_set_anisotropy(&g, (anisotropy){.ani = 0.02 * QE * 1.0e-3, .dir = v3d_c(0.0, 0.0, 1.0)});

    for (int r = 0; r < g.gi.rows; ++r)
        for (int c = g.gi.cols / 2.0; c < g.gi.cols; ++c)
            grid_set_anisotropy_loc(&g, r, c, (anisotropy){.ani = 0.05 * QE * 1.0e-3, .dir = v3d_c(0.0, 0.0, 1.0)});

    string_view current_func = sv_from_cstr("current ret = (current){0};\n"\
                                                                   "ret.type = CUR_STT;\n"\
                                                                   "time -= 0.1 * NS;\n"\
                                                                   "double j_ac = 5.0e10 * (time > 0);\n"\
                                                                   "double j_dc = 1.0e10 * (time > 0);\n"\
                                                                   "double omega = 319145920.365;\n"\
                                                                   "ret.stt.j = v3d_c(j_dc, j_ac * sin(omega * time), 0.0);\n"\
                                                                   "ret.stt.polarization = -1.0;\n"\
                                                                   "ret.stt.beta = 0.0;\n"\
                                                                   "return (current){0};\n"\
                                                                   "return ret;");
    string_view field_func = sv_from_cstr("double normalized = 0.5;\n"\
                                          "double real = normalized * gs.dm * gs.dm / gs.exchange / gs.mu;\n"\
                                          "double osc = sin(5 * M_PI * gs.col / 64.0 - M_PI * 1.0 * time / NS);\n"\
                                          "real = real * (1.0 + 1 * osc);\n"\
                                          "return v3d_c(0.0, 0.0, real);");
    string_view compile = sv_from_cstr("-cl-fast-relaxed-math");
    /*profiler_start_measure("Integration");
    integrate(&g, .dt = dt, .duration = 10.0 * NS, .current_generation_function = current_func, .field_generation_function = field_func, .compile_augment = compile, .interval_for_information=1519268);
    profiler_end_measure("Integration");
    profiler_print_measures(stdout);
    return 0;*/

    grid_renderer gr = grid_renderer_init(&g, window, current_func, field_func, (string_view){0}, compile);
    integrate_context ctx = integrate_context_init(&g, &gr.gpu, dt);

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

    integrate_context_close(&ctx);
    grid_renderer_close(&gr);
    grid_free(&g);
    window_close(window);
    return 0;
}
