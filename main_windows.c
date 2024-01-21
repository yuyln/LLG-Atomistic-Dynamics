#include "atomistic_simulation.h"
#include <time.h>

void run_gsa(grid *g, gpu_cl *gpu) {
    double ratio = (double)g->gi.cols / g->gi.rows;
    window_init("GSA", 800 * ratio, 800);

    grid_renderer gr = grid_renderer_init(g, gpu);
    gsa_context ctx = gsa_context_init(g, gr.gpu, .T0 = 500.0, .inner_steps=700000, .qV = 2.7, .print_factor=10);
    int state = 'h';

    const int steps = 100;

    while(!window_should_close()) {
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
    }
    gsa_context_read_minimun_grid(&ctx);
    gsa_context_close(&ctx);
    grid_renderer_close(&gr);
    //window_close(window);
}

void run_integration(grid *g, gpu_cl *gpu, double dt) {
    double ratio = (double)g->gi.cols / g->gi.rows;
    window_init("Integration", 800 * ratio, 800);

    grid_renderer gr = grid_renderer_init(g, gpu);
    integrate_context ctx = integrate_context_init(g, gpu, dt);

    int state = 'h';

    double time_for_print = 1.0;
    double stopwatch_print = -1.0;
    int frames = 0;
    const int steps = 100;

    while(!window_should_close()) {
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

    }
    integrate_context_close(&ctx);
    grid_renderer_close(&gr);
    //window_close(window);
}

void run_gradient_descent(grid *g, gpu_cl *gpu, double dt) {
    double ratio = (double)g->gi.cols / g->gi.rows;
    window_init("Gradient Descent", 800 * ratio, 800);

    grid_renderer gr = grid_renderer_init(g, gpu);
    gradient_descent_context ctx = gradient_descent_context_init(g, gr.gpu, .dt=dt, .T = 500.0, .T_factor = 0.9999);

    int state = 'h';

    double time_for_print = 1.0;
    double stopwatch_print = -1.0;
    int frames = 0;
    const int steps = 100;

    while(!window_should_close()) {
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

        window_poll();
        window_render();

    }
    gradient_descent_read_mininum_grid(&ctx);
    gradient_descent_close(&ctx);
    grid_renderer_close(&gr);
    //window_close(window);
}

int main(void) {
    int rows = 128;
    int cols = 128;
    double dt = HBAR / (1.0e-3 * QE) * 0.01;

    grid g = grid_init(rows, cols);

    grid_set_dm(&g, 1.00 * QE * 1.0e-3, 0.0, R_ij);
    grid_set_anisotropy(&g, (anisotropy){.ani = 0.0 * QE * 1.0e-3, .dir = v3d_c(0.0, 0.0, 1.0)});
    v3d_fill_with_random(g.m, rows, cols);

    string_view current_func = sv_from_cstr("current ret = (current){0};\n"\
                                             "return ret;\n"\
                                             "ret.type = CUR_SHE;\n"\
                                             "time -= 0.1 * NS;\n"\
                                             "ret.she.p = v3d_c(0.0, 1.0e10, 0.0);\n"\
                                             "ret.she.thickness = gs.lattice;\n"\
                                             "ret.she.beta = 0.0;\n"\
                                             "ret.she.theta_sh = 1.0;\n"\
                                             "return ret;");
    string_view field_func = sv_from_cstr("double normalized = 0.5;\n"\
                                          "double real = normalized * gs.dm * gs.dm / gs.exchange / gs.mu;\n"\
                                          "//double osc = sin(5 * M_PI * gs.col / 64.0 - M_PI * 1.0 * time / NS);\n"\
                                          "//real = real * (1.0 + 0.1 * osc);\n"\
                                          "return v3d_c(0.0, 0.0, real);");

    string_view temperature_func = sv_from_cstr("return 0.0 / (time / NS + EPS);");

    string_view compile = sv_from_cstr("-cl-fast-relaxed-math");

    //integrate(&g, .dt = dt, .duration = 1 * NS, .current_generation_function = current_func, .field_generation_function = field_func, .compile_augment = compile);

    srand(time(NULL));

    gpu_cl gpu = gpu_cl_init(current_func, field_func, temperature_func, sv_from_cstr(""), compile);
    run_gsa(&g, &gpu);
    run_gradient_descent(&g, &gpu, 1.0e-1);
    run_integration(&g, &gpu, dt);

    grid_free(&g);
    return 0;
}
