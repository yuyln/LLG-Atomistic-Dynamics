#include "atomistic_simulation.h"

void set_pin(grid *g, uint64_t row, uint64_t col, void *dummy) {
    UNUSED(dummy);
    g->gp[row * g->gi.cols + col].pin = (pinning){.pinned = 1, .dir = v3d_c(0, 0, -1)};
}

//@TODO: PROPER ERROR CHECKING URGENT!!!
//@TODO: check time measuring when compiling it to win
int main(void) {
    unsigned int rows = 64;
    unsigned int cols = 64;

    double lattice = 0.5e-9;
    double alpha = 0.3;
    double J = 1.0e-3 * QE;
    double dm = 0.5 * J;
    double ani = 0.05 * J;

    grid g = grid_init(rows, cols);
    g.gp[0].pin = (pinning){.pinned = 1, .dir = v3d_c(1, 0, 0)};
    grid_set_lattice(&g, lattice);
    grid_set_alpha(&g, alpha);
    grid_set_exchange(&g, J);
    grid_set_anisotropy(&g, (anisotropy){.dir=v3d_c(0.0, 0.0, 1.0), .ani = ani});
    double mu = g.gp->mu;
    dm_interaction default_dm = (dm_interaction){.dmv_down = v3d_c(dm, 0.0, 0.0),
                                                 .dmv_up = v3d_c(-dm, 0.0, 0.0),
                                                 .dmv_left = v3d_c(0.0, -dm, 0.0),
                                                 .dmv_right = v3d_c(0.0, dm, 0.0)};

    for (unsigned int i = 0; i < rows * cols; ++i)
        g.m[i] = v3d_c(0, 0, -1);
    v3d_create_skyrmion(g.m, rows, cols, 10, rows / 2, 2.3 * cols / 5, 1, -1, M_PI / 2);

    int a = 8;
    grid_do_in_ellipse(&g, cols / 2 + 10, a, a, a, set_pin, NULL);

    grid_set_dm(&g, default_dm);
    double dt = 0.01 * HBAR / (J * SIGN(J));
    string current_func = str_is_cstr("current ret = (current){};\n"\
                                      "time -= 0.5 * NS;\n"\
                                      "ret.type = CUR_STT;\n"\
                                      "ret.stt.j = v3d_c(0.0, -20.0e10, 0.0);\n"\
                                      "ret.stt.beta = 0.0;\n"\
                                      "ret.stt.polarization = -1.0;\n"\
                                      "ret.stt.j = v3d_scalar(ret.stt.j, (time > 0));\n"\
                                      "return ret;");

    string field_func = str_from_fmt("double Hz = -%.15e;\n"\
                                     "return v3d_c(0.0, 0.0, Hz);", 0.5 * dm * dm / (J * SIGN(J)) * 1.0 / mu);

    string temperature_func = str_is_cstr("return 0.0;");
    string compile = str_is_cstr("-cl-fast-relaxed-math");

    double ratio = (double)rows / cols;

    gradient_descent_params gd_params = gradient_descent_params_init();
    UNUSED(gd_params);
    gd_params.dt = 5.0e-3;
    gd_params.T = 500;
    gd_params.T_factor = 0.99998918028024917967;
    gd_params.compile_augment = compile;
    gd_params.field_func = field_func;

    logging_log(LOG_INFO, "Integration dt: %e", dt);
    integrate_params int_params = integrate_params_init();
    int_params.current_func = current_func;
    int_params.field_func = field_func;
    int_params.temperature_func = temperature_func;
    int_params.compile_augment = compile;
    int_params.duration = 20 * NS;
    int_params.interval_for_raw_grid = 1000;
    grid_renderer_integrate(&g, int_params, 400 / ratio, 400);

    str_free(&field_func);
    grid_free(&g);
    return 0;
}
