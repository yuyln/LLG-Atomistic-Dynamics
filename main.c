#include "atomistic_simulation.h"


//@TODO: PROPER ERROR CHECKING URGENT!!!
//@TODO: check time measuring when compiling it to win
int Stosic(void) {
    unsigned int rows = 64;
    unsigned int cols = 64;

    double lattice = 0.251e-9;
    double alpha = 0.037;
    double J = 29.0e-3 * QE;
    double dm = 1.5e-3 * QE;
    double ani = 0.293e-3 * QE;

    grid g = grid_init(rows, cols);
    grid_set_lattice(&g, lattice);
    grid_set_alpha(&g, alpha);
    grid_set_exchange(&g, J);
    grid_set_anisotropy(&g, (anisotropy){.dir=v3d_c(0.0, 0.0, 1.0), .ani = ani});
    double mu = 2.1 * MU_B;
    grid_set_mu(&g, mu);
    dm_interaction default_dm = (dm_interaction){.dmv_down = v3d_c(dm, 0.0, 0.0),
                                                 .dmv_up = v3d_c(-dm, 0.0, 0.0),
                                                 .dmv_left = v3d_c(0.0, -dm, 0.0),
                                                 .dmv_right = v3d_c(0.0, dm, 0.0)};

    grid_set_dm(&g, default_dm);
    for (unsigned int i = 0; i < rows * cols; ++i)
        g.m[i] = v3d_c(0, 0, -1);


    v3d_create_skyrmion(g.m, g.gi.rows, g.gi.cols, 10, rows / 2, cols / 2, 1, 1, M_PI / 2.0);
    g.gi.pbc = (pbc_rules){.pbc_x = 0, .pbc_y = 0, .m = v3d_c(0, 0, -1)};

    double dt = 0.01 * HBAR / (J * SIGN(J));
    //string current_func = str_is_cstr("current ret = (current){};\n"\
    //                                  "time -= 0.002 * NS;\n"\
    //                                  "ret.type = CUR_STT;\n"\
    //                                  "ret.stt.j = v3d_c(1e10, 10e10 * sin(time / NS * 5), 0.0);\n"\
    //                                  "ret.stt.beta = 0.0;\n"\
    //                                  "ret.stt.polarization = -1.0;\n"\
    //                                  "ret.stt.j = v3d_scalar(ret.stt.j, (time > 0));\n"\
    //                                  "return ret;");

    //string field_func = str_from_fmt("double Hz = %.15e;\n"\
    //                                 "return v3d_c(0.0, 0.0, Hz);", 0.5 * dm * dm / J * 1.0 / mu);



    logging_log(LOG_INFO, "Integration dt: %e", dt);
    integrate_params int_params = integrate_params_init();
    int_params.field_func = create_field_D2_over_J(v3d_c(0, 0, 0.5), J, dm, mu);
    int_params.duration = 20 * NS;
    int_params.interval_for_raw_grid = 1000;
    int_params.dt = dt;
    grid_renderer_integrate(&g, int_params, 1000, 1000);

    //str_free(&field_func);
    grid_free(&g);
    return 0;
}

struct data {
    double dm, J, ani;
    int size;
};

void set_pin(grid *g, uint64_t row, uint64_t col, void *dummy) {
    struct data *it = (struct data*)dummy;
    int center = 32;
    int dx = col - center;
    int dy = row - center;
    double d2 = dx * dx + dy * dy;
    grid_set_exchange_loc(g, row, col, it->J * 2);
    double angle = atan2(dy, dx) + M_PI;
    g->m[row * g->gi.cols + col] = v3d_c(0 * cos(angle), 0 * sin(angle), -1);
}

void set_pin2(grid *g, uint64_t row, uint64_t col, void *dummy) {
    UNUSED(dummy);
    g->m[row * g->gi.cols + col] = v3d_c(0, 0, -1);
}

int test(void) {
    unsigned int rows = 128;
    unsigned int cols = 128;

    //double lattice = 0.5e-9;
    //double alpha = 0.4;
    //double J = 1.0e-3 * QE;
    //double dm = 0.5 * J;
    //double ani = 0.05 * J;
    double lattice = 0.5e-9;
    double alpha = 0.04;
    double J = 1.0e-3 * QE;
    double dm = 0.2 * J;
    double ani = 0.01 * J;

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

    grid_set_dm(&g, default_dm);

    for (unsigned int i = 0; i < rows * cols; ++i)
        g.m[i] = v3d_c(0, 0, -1);

    int n = 6;

    double Ry = 2.0 * rows / (3.0 * n);
    double Rx = 2.0 * cols / (3.0 * n);

    for (unsigned int iy = 0; iy < n; ++iy) {
        int yc = Ry / 4.0 + iy * (Ry + Ry / 2.0) + Ry / 2.0;
        for (unsigned int ix = 0; ix < n; ++ix) {
            int xc = Rx / 4.0 + ix * (Rx + Rx / 2.0) + Rx / 2.0;
            if (iy % 2 == 0)
                xc += Rx / 2.0 + Rx / 4.0;
            //v3d_create_skyrmion(g.m, g.gi.rows, g.gi.cols, Rx / 2, yc, xc, 1, 1, M_PI / 2.0);
        }
    }
    v3d_create_skyrmion(g.m, g.gi.rows, g.gi.cols, 3 * Rx, rows / 2, 2.5 * cols / 5, 1, 1, M_PI / 2.0);
    v3d_create_skyrmion(g.m, g.gi.rows, g.gi.cols, Rx, rows / 2, 2.5 * cols / 5, -1, -1, M_PI / 2.0);


    int a = cols / 4;
    struct data user_data = {.ani = ani, .dm = dm, .J = J, .size = a};
    //grid_do_in_ellipse(&g, cols / 2 , rows / 2, 1 * a, 2 * a, set_pin2, NULL);
    //grid_do_in_ellipse(&g, cols / 2 , rows / 2, a, a, set_pin, &user_data);

    double dt = 0.01 * HBAR / (J * SIGN(J));
    //string current_func = str_is_cstr("current ret = (current){};\n"\
    //                                  "time -= 0.2 * NS;\n"\
    //                                  "ret.type = CUR_STT;\n"\
    //                                  "ret.stt.j = v3d_c(0.2e10 * cos(time / (NS * 40)), 0.2e10 * sin(time / (NS * 40)), 0.0);\n"\
    //                                  "ret.stt.beta = 0.0;\n"\
    //                                  "ret.stt.polarization = -1.0;\n"\
    //                                  "ret.stt.j = v3d_scalar(ret.stt.j, (time > 0));\n"\
    //                                  "return ret;");

    //string field_func = str_from_fmt("double Hz = -%.15e;\n"\
    //                                 "return v3d_c(0.0, 0.0, Hz);", 0.5 * dm * dm / J * 1.0 / mu);

    //string temperature_func = str_is_cstr("return 0.0;");
    //string compile = str_is_cstr("-cl-fast-relaxed-math");

    double ratio = (double)rows / cols;

    logging_log(LOG_INFO, "Integration dt: %e", dt);
    integrate_params int_params = integrate_params_init();
    //int_params.current_func = create_current_she_ac(20e10, v3d_c(1, 1, 0), 20.0 / NS, 0);
    //int_params.current_func = create_current_stt_dc(20e10, 0, 0);
    //int_params.current_func = create_current_stt_dc_ac(20e10, 0, 0, 20e10, 20.0 / NS, 0);
    int_params.field_func = create_field_D2_over_J(v3d_c(0, 0, -0.5), J, dm, mu);
    int_params.duration = 200.2 * NS;
    int_params.interval_for_raw_grid = 0;
    int_params.dt = dt * 1;
    int_params.interval_for_raw_grid = 10000;
    grid_renderer_integrate(&g, int_params, 1000, 1000);
    int_params.current_func = create_current_stt_dc(5e10, 0, 0);
    //int_params.current_func = create_current_she_dc(2e10, v3d_c(1, 0, 0), 0);
    grid_renderer_integrate(&g, int_params, 1000, 1000);

    //str_free(&field_func);
    grid_free(&g);
    return 0;

}

int main(void) {
    return test();
}

