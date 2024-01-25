#define __PROFILER_IMPLEMENTATION
#include "atomistic_simulation.h"
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define steps 100

void run_gsa(grid *g, gpu_cl *gpu) {
    double ratio = (double)g->gi.cols / g->gi.rows;
    window_init("GSA", 800 * ratio, 800);

    grid_renderer gr = grid_renderer_init(g, gpu);
    gsa_context ctx = gsa_context_init(g, gr.gpu, .T0 = 10.0,
            .inner_steps = 700000, .print_factor = 10);

    struct timespec current_time;
    clock_gettime(CLOCK_REALTIME, &current_time);
    int state = 'h';

    double time_for_print = 1.0;
    double stopwatch_print = -1.0;
    int frames = 0;

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

        for (int i = 0; i < steps; ++i) {
            gsa_metropolis_step(&ctx);
            gsa_thermal_step(&ctx);
        }

        window_render();
        window_poll();

        struct timespec new_time;
        clock_gettime(CLOCK_REALTIME, &new_time);
        double dt_real = (new_time.tv_sec + new_time.tv_nsec * 1.0e-9) -
            (current_time.tv_sec + current_time.tv_nsec * 1.0e-9);
        current_time = new_time;
        stopwatch_print += dt_real;
        frames++;
        if (stopwatch_print >= 0) {
            logging_log(LOG_INFO, "FPS: %d - Frame Time: %e ms",
                    (int)(frames / time_for_print),
                    time_for_print / frames / 1.0e-3);
            stopwatch_print = -1.0;
            frames = 0;
        }
    }
    gsa_context_read_minimun_grid(&ctx);
    gsa_context_close(&ctx);
    grid_renderer_close(&gr);
}

void run_integration(grid *g, gpu_cl *gpu, double dt) {
    double ratio = (double)g->gi.cols / g->gi.rows;
    window_init("Integration", 800 * ratio, 800);

    grid_renderer gr = grid_renderer_init(g, gpu);
    integrate_context ctx = integrate_context_init(g, gpu, dt);

    struct timespec current_time;
    clock_gettime(CLOCK_REALTIME, &current_time);
    int state = 'b';

    double time_for_print = 1.0;
    double stopwatch_print = -1.0;
    int frames = 0;

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

        for (int i = 0; i < steps; ++i) {
            integrate_step(&ctx);
            integrate_exchange_grids(&ctx);
            ctx.time += dt;
        }

        window_render();
        window_poll();

        struct timespec new_time;
        clock_gettime(CLOCK_REALTIME, &new_time);
        double dt_real = (new_time.tv_sec + new_time.tv_nsec * 1.0e-9) -
            (current_time.tv_sec + current_time.tv_nsec * 1.0e-9);
        current_time = new_time;
        stopwatch_print += dt_real;
        frames++;
        if (stopwatch_print >= 0) {
            logging_log(LOG_INFO,
                    "FPS: %d - Frame Time: %e ms - System Steps: %d - System dt: "
                    "%e - System Time: %e",
                    (int)(frames / time_for_print),
                    time_for_print / frames / 1.0e-3, steps * frames,
                    frames * steps * dt, ctx.time);
            stopwatch_print = -1.0;
            frames = 0;
        }
    }
    integrate_context_close(&ctx);
    grid_renderer_close(&gr);
}

void run_gradient_descent(grid *g, gpu_cl *gpu, double dt) {
    double ratio = (double)g->gi.cols / g->gi.rows;
    window_init("Gradient Descent", 800 * ratio, 800);

    grid_renderer gr = grid_renderer_init(g, gpu);
    gradient_descent_context ctx = gradient_descent_context_init(
            g, gr.gpu, .dt = dt, .T = 500.0, .T_factor = 0.99999);

    struct timespec current_time;
    clock_gettime(CLOCK_REALTIME, &current_time);
    int state = 'h';

    double time_for_print = 1.0;
    double stopwatch_print = -1.0;
    int frames = 0;

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

        for (int i = 0; i < steps; ++i) {
            gradient_descent_step(&ctx);
            gradient_descent_exchange(&ctx);
        }

        window_render();
        window_poll();

        struct timespec new_time;
        clock_gettime(CLOCK_REALTIME, &new_time);
        double dt_real = (new_time.tv_sec + new_time.tv_nsec * 1.0e-9) -
            (current_time.tv_sec + current_time.tv_nsec * 1.0e-9);
        current_time = new_time;
        stopwatch_print += dt_real;
        frames++;
        if (stopwatch_print >= 0) {
            logging_log(LOG_INFO,
                    "FPS: %d - Frame Time: %e ms - System Steps: %d - System dt: "
                    "%e - System Temperature %e - System Mininum Energy: %e",
                    (int)(frames / time_for_print),
                    time_for_print / frames / 1.0e-3, steps * frames,
                    frames * steps * dt, ctx.T, ctx.min_energy);
            stopwatch_print = -1.0;
            frames = 0;
        }
    }
    gradient_descent_read_mininum_grid(&ctx);
    gradient_descent_close(&ctx);
    grid_renderer_close(&gr);
}

//@TODO: Do 3D
//@TODO: Clear everything on integrate context and gsa context(done?)
//@TODO: Proper error handling
//@TODO: My create buffer should not be a function from gpu_cl. This difficults
//figuring where the error came from
//@TODO: Change things on gpu.h to macros to print file and line
//@TODO: Create functions to better get DM vectors
int main(void) {
    int rows = 128;
    int cols = 128;

    grid g = grid_init(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            g.m[r * cols + c] = v3d_c(0.0, 0.0, 1.0);

    double J = 1.0e-3 * QE;

    double dm = 0.2 * J;
    dm_interaction default_dm = (dm_interaction){.dmv_down = v3d_c(0.0, -dm, 0.0),
                                                 .dmv_up = v3d_c(0.0, dm, 0.0),
                                                 .dmv_left = v3d_c(-dm, 0.0, 0),
                                                 .dmv_right = v3d_c(dm, 0.0, 0)};

    grid_set_alpha(&g, 0.1);
    grid_set_dm(&g, default_dm);
    grid_set_anisotropy(&g, (anisotropy){.ani = 0.02 * J, .dir = v3d_c(0.0, 0.0, 1.0)});
    grid_set_mu(&g, HBAR * g.gp->gamma);
    grid_set_lattice(&g, 0.5e-9);

    v3d_create_skyrmion(g.m, g.gi.rows, g.gi.cols, 30, rows / 2.0, cols / 2.0, -1.0, 1.0, 0.0);
    v3d_create_skyrmion(g.m, g.gi.rows, g.gi.cols, 10, rows / 2.0, cols / 2.0, 1.0, -1.0, 0.0);
    J *= 1.0;
    grid_set_exchange(&g, J);
    for (int r = 0; r < g.gi.rows; ++r) {
        for (int c = 0; c < g.gi.cols; ++c) {
            v3d m = g.m[r * g.gi.cols + c];
            double x = m.x;
            double y = m.y;
            double z = m.z;
            g.m[r * g.gi.cols + c] = v3d_scalar(v3d_c(x, y, z), pow(SIGN(J), r + c));

        }
    }
    grid_set_pinning_loc(&g, rows / 2, cols / 5, (pinning){.pinned=1, .dir=v3d_c(0.0, 0.0, 1.0)});
    grid_set_pinning_loc(&g, rows / 2 + 1, cols / 5, (pinning){.pinned=1, .dir=v3d_c(0.0, 0.0, 1.0)});
    grid_set_pinning_loc(&g, rows / 2, cols / 5 + 1, (pinning){.pinned=1, .dir=v3d_c(0.0, 0.0, 1.0)});
    grid_set_pinning_loc(&g, rows / 2 + 1, cols / 5 + 1, (pinning){.pinned=1, .dir=v3d_c(0.0, 0.0, 1.0)});


    double dt = 0.01 * HBAR / (J * SIGN(J));

    string current_func = str_is_cstr("current ret = (current){};\n"\
                                      "//return ret;\n"\
                                      "ret.type = CUR_SHE;\n"\
                                      "ret.she.p = v3d_c(1.0e9 * (time > 1 * NS), 0.0, 0.0);\n"\
                                      "ret.she.beta = 1.0;\n"\
                                      "ret.she.theta_sh = -1.0;\n"\
                                      "ret.she.thickness = gs.lattice;\n"\
                                      "return ret;");

    string field_func = str_from_fmt("double Hz = %.15e / gs.mu;\n"\
                                     "double Hy = 0.004 * gs.exchange / gs.mu;\n"\
                                     "double w = 0.017 * gs.exchange / HBAR;\n"\
                                     "double h = 2.0e-4 * sin(w * time) * gs.exchange / gs.mu;\n"\
                                     "return v3d_c(0.0, 0.0, Hz);", 0.405 * dm * dm / (J * SIGN(J)));

    string temperature_func = str_is_cstr("return 0.0;");

    string compile = str_is_cstr("-cl-fast-relaxed-math");

    srand(time(NULL));

    gpu_cl gpu = gpu_cl_init(current_func, field_func, temperature_func, STR_NULL, compile);
    str_free(&field_func);
    logging_log(LOG_INFO, "Integration dt: %e", dt);
    //run_gsa(&g, &gpu);
    //run_gradient_descent(&g, &gpu, 1.0e-1);
    run_integration(&g, &gpu, dt * 1.0);
    grid_free(&g);
    return 0;
}
