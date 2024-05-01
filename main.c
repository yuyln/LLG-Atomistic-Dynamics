#include "atomistic_simulation.h"

void set_pin(grid *g, uint64_t row, uint64_t col) {
    g->gp[row * g->gi.cols + col].pin = (pinning){.pinned = 1, .dir = v3d_c(0, 0, -1)};
}

//@TODO: PROPER ERROR CHECKING URGENT!!!
//@TODO: check time measuring when compiling it to win
int main(void) {
    unsigned int rows = 64;
    unsigned int cols = 64;

    double lattice = 0.5e-9;
    double alpha = 0.04;
    double J = 1.0e-3 * QE;
    double dm = 1.0 * J;
    double ani = 0.05 * J;

    grid g = grid_init(rows, cols);
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
    v3d_create_skyrmion(g.m, rows, cols, 7, rows / 2, 4 * cols / 5, 1, -1, M_PI / 2);

    //for (unsigned int i = 0; i < rows; ++i)
    //    g.gp[i * cols + (cols - 1)].pin = (pinning){.dir = v3d_c(0, 0, -1), .pinned = 1};
    int a = 8;
    //grid_do_in_ellipse(&g, cols / 2 + 15, a, a, a, set_pin);
    //grid_do_in_rect(&g, cols / 2, 2 * a, cols / 2 + a, 3 * a, set_pin);
    grid_do_in_line(&g, 0, 0, cols, rows, 5, set_pin);

    grid_set_dm(&g, default_dm);
    double dt = 0.01 * HBAR / (J * SIGN(J));
    string current_func = str_is_cstr("current ret = (current){};\n"\
                                      "time -= 0.5 * NS;\n"\
                                      "ret.type = CUR_STT;\n"\
                                      "ret.stt.j = v3d_c(0.0, -5.0e10, 0.0);\n"\
                                      "ret.stt.beta = 0.0;\n"\
                                      "ret.stt.polarization = -1.0;\n"\
                                      "ret.stt.j = v3d_scalar(ret.stt.j, (time > 0));\n"\
                                      "return ret;");

    //string current_func = str_is_cstr("current ret = (current){};\n"\
                                      "time -= 0.5 * NS;\n"\
                                      "ret.type = CUR_SHE;\n"\
                                      "ret.she.p = v3d_c(-2.5e10, 0.0, 0.0);\n"\
                                      "ret.she.beta = 0.0;\n"\
                                      "ret.she.theta_sh = -1.0;\n"\
                                      "ret.she.thickness = 0.5 * NANO;\n"\
                                      "ret.she.p = v3d_scalar(ret.she.p, (time > 0));\n"\
                                      "return ret;");

    string field_func = str_from_fmt("double Hz = -%.15e;\n"\
                                     "return v3d_c(0.0, 0.0, Hz);", 0.5 * dm * dm / (J * SIGN(J)) * 1.0 / mu);

    string temperature_func = str_is_cstr("return 0.0;");
    string compile = str_is_cstr("-cl-fast-relaxed-math");

    double ratio = (double)rows / cols;

    gradient_descent_params gd_params = gradient_descent_params_init();
    gd_params.dt = 5.0e-3;
    //gd_params.T = 5000.0;
    gd_params.T = 500;
    gd_params.T_factor = 0.99998918028024917967;
    gd_params.compile_augment = compile;
    gd_params.field_func = field_func;
    //gd_params.damping = 1.0;
    //gd_params.restoring = 10.0;
    //gd_params.steps = 100000;
    //steps_per_frame = 100;
    //grid_renderer_gradient_descent(&g, gd_params, 400 / ratio, 400);
    //gradient_descent(&g, gd_params);

    logging_log(LOG_INFO, "Integration dt: %e", dt);
    integrate_params int_params = integrate_params_init();
    int_params.dt = dt;
    int_params.current_func = current_func;
    int_params.field_func = field_func;
    int_params.temperature_func = temperature_func;
    int_params.compile_augment = compile;
    int_params.duration = 20 * NS;
    grid_renderer_integrate(&g, int_params, 800 / ratio, 800);
    //integrate(&g, int_params);

    str_mfree(&field_func);
    grid_mfree(&g);
    return 0;
}
