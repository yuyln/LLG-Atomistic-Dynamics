#define __PROFILER_IMPLEMENTATION
#include "atomistic_simulation.h"

void set_pin(grid *g, uint64_t row, uint64_t col) {
    g->gp[row * g->gi.cols + col].pin = (pinning){.pinned = 1, .dir = v3d_c(0, 0, -1)};
}

//@TODO: PROPER ERROR CHECKING URGENT!!!
int main(void) {
    unsigned int rows = 272 / 2;
    unsigned int cols = 272 * 2;

    double lattice = 0.5e-9;
    double alpha = 0.3;
    double J = 1.0e-3 * QE;
    double dm = 0.18 * J;
    double ani = 0.00 * J;

    grid g = grid_init(rows, cols);
    grid_set_lattice(&g, lattice);
    grid_set_alpha(&g, alpha);
    grid_set_exchange(&g, J);
    grid_set_anisotropy(&g, (anisotropy){.dir=v3d_c(0.0, 0.0, 1.0), .ani = ani});
    double mu = g.gp->mu;
    grid_do_in_rect(&g, 0, 0, cols, 10, set_pin);
    grid_do_in_rect(&g, 0, rows - 10, cols, rows, set_pin);

    for (uint64_t i = 0; i < 10; ++i)
        grid_do_in_line(&g, cols * i / 10.0, i % 2? 0: rows, cols * (i + 1) / 10.0, rows / 2, 4, set_pin);

    dm_interaction default_dm = (dm_interaction){.dmv_down = v3d_c(-dm, 0.0, 0.0),
                                                 .dmv_up = v3d_c(dm, 0.0, 0.0),
                                                 .dmv_left = v3d_c(0.0, dm, 0.0),
                                                 .dmv_right = v3d_c(0.0, -dm, 0.0)};

    grid_set_dm(&g, default_dm);
    double dt = 0.01 * HBAR / (J * SIGN(J));
    string current_func = str_is_cstr("current ret = (current){};\n"\
                                      "time -= 0.5 * NS;\n"\
                                      "ret.type = CUR_STT;\n"\
                                      "ret.stt.j = v3d_c(0.0, -2.05e10, 0.0);\n"\
                                      "ret.stt.beta = 0.0;\n"\
                                      "ret.stt.polarization = -1.0;\n"\
                                      "ret.stt.j = v3d_scalar(ret.stt.j, time > 0);\n"\
                                      "return ret;");

    string field_func = str_from_fmt("double Hz = -%.15e;\n"\
                                     "return v3d_c(0.0, 0.0, Hz);", 0.5 * dm * dm / (J * SIGN(J)) * 1.0 / mu);

    string temperature_func = str_is_cstr("return 0.0;");
    string compile = str_is_cstr("-cl-fast-relaxed-math");

    srand(time(NULL));
    double ratio = (double)rows / cols;

    gradient_descent_params gd_params = gradient_descent_params_init();
    gd_params.dt = 1.0e-2;
    gd_params.T = 5000.0;
    gd_params.T_factor = 0.99985;
    gd_params.compile_augment = compile;
    gd_params.field_func = field_func;
    gd_params.damping = 1.0;
    gd_params.restoring = 10.0;
    gd_params.steps = 100000;
    profiler_start_measure("test");
    grid_renderer_gradient_descent(&g, gd_params, 400 / ratio, 400);
    profiler_end_measure("test");
    profiler_print_measures(stdout);
    //gradient_descent(&g, gd_params);

    logging_log(LOG_INFO, "Integration dt: %e", dt);
    integrate_params int_params = integrate_params_init();
    int_params.dt = dt;
    int_params.current_func = current_func;
    int_params.field_func = field_func;
    int_params.temperature_func = temperature_func;
    int_params.compile_augment = compile;
    grid_renderer_integrate(&g, int_params, 400 / ratio, 400);
    //integrate(&g, int_params);

    str_free(&field_func);
    grid_free(&g);
    return 0;
}
