#include "atomistic_simulation.h"
#include <stdint.h>
#include <float.h>

//@TODO: PROPER ERROR CHECKING URGENT!!!
//@TODO: check time measuring when compiling it to windows
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


    grid_create_skyrmion_at(&g, 10, 5, cols / 2, rows / 2, 1, 1, 0);
    g.gi.pbc = (pbc_rules){.pbc_x = 0, .pbc_y = 0, .m = v3d_c(0, 0, -1)};

    double dt = 0.01 * HBAR / (J * SIGN(J));

    logging_log(LOG_INFO, "Integration dt: %e", dt);
    integrate_params int_params = integrate_params_init();
    int_params.field_func = create_field_D2_over_J(v3d_c(0, 0, 0.5), J, dm, mu);
    int_params.duration = 20 * NS;
    int_params.interval_for_raw_grid = 1000;
    int_params.dt = dt;
    grid_renderer_integrate(&g, int_params, 1000, 1000);

    grid_free(&g);
    return 0;
}

int skyrmionium_testing(void) {
    unsigned int rows = 64;
    unsigned int cols = 64;

    double lattice = 0.5e-9;
    double J = 1.0e-3 * QE;
    double dm = 0.2 * J;
    double ani = 0.01 * J;
    double alpha = 0.3;

    grid g = grid_init(rows, cols);
    grid_set_lattice(&g, lattice);
    grid_set_alpha(&g, alpha);
    grid_set_exchange(&g, J);
    grid_set_anisotropy(&g, (anisotropy){.dir=v3d_c(0.0, 0.0, 1.0), .ani = ani});

    double mu = g.gp->mu;

    dm_interaction default_dm = dm_interfacial(dm);

    grid_set_dm(&g, default_dm);

    grid_uniform(&g, v3d_c(0, 0, 1));


    double dt = 0.01 * HBAR / (J * SIGN(J));
    double ratio = (double)rows / cols;
    logging_log(LOG_INFO, "Integration dt: %e", dt);
    integrate_params int_params = integrate_params_init();
    int_params.field_func = create_field_D2_over_J(v3d_c(0, 0, 0.5), J, dm, mu);
    int_params.duration = 200.2 * NS;
    int_params.interval_for_raw_grid = 1000;
    int_params.dt = dt;
    int_params.do_cluster = true;
    int_params.interval_for_cluster = 100;

    grid_create_skyrmionium_at(&g, 10, 10, cols / 2.0, rows / 2.0, 1, 1, 0);
    //int_params.current_func = str_is_cstr("current ret = (current){.type=CUR_SHE};\n"\
    //                                      "double x = gs.col - 0.5 * 64;\n"\
    //                                      "double y = gs.row - 0.5 * 64;\n"\
    //                                      "int ring = x * x + y * y <= 15 * 15 && x * x + y * y >= 13 * 13;\n"\
    //                                      "ret.she.p = v3d_scalar(v3d_normalize(v3d_c(0, .1, -1)), 1000e10 * ring * (time < 40e-12));\n"\
    //                                      "ret.she.thickness = 0.5e-9;\n"\
    //                                      "ret.she.beta = 0;\n"\
    //                                      "ret.she.theta_sh = 1;\n"\
    //                                      "return ret;\n");

    grid_renderer_integrate(&g, int_params, 1000, 1000);

    double jx = 1e10;
    double jy = 1e10;
    int_params.current_func = create_current_she_ac(jx, v3d_c(jx, jy, 0), 100 / (200 * NS), 0);
    grid_renderer_integrate(&g, int_params, 1000, 1000);

    grid_free(&g);
    return 0;
}

int ratchet_testing(void) {
    unsigned int rows = 64;
    unsigned int cols = 64;

    double lattice = 0.5e-9;
    double J = 1.0e-3 * QE;
    double dm = 0.2 * J;
    double ani = 0.02 * J;
    double alpha = 0.3;

    grid g = grid_init(rows, cols);

    double mu = g.gp->mu;
    grid_set_lattice(&g, lattice);
    grid_set_alpha(&g, alpha);
    grid_set_exchange(&g, J);
    grid_set_anisotropy(&g, (anisotropy){.dir=v3d_c(0.0, 0.0, 1.0), .ani = ani});
    grid_set_mu(&g, mu);

    dm_interaction default_dm = dm_interfacial(-dm);
    grid_set_dm(&g, default_dm);


    double dt = 0.01 * HBAR / (J * SIGN(J));
    double ratio = (double)rows / cols;

    logging_log(LOG_INFO, "Integration dt: %e", dt);
    
    integrate_params int_params = integrate_params_init();
    int_params.field_func = create_field_D2_over_J(v3d_c(0, 0, 0.1), J, dm, mu);
    int_params.duration = 200.2 * NS;
    int_params.interval_for_raw_grid = 1000;
    int_params.dt = dt;
    int_params.do_cluster = true;
    int_params.interval_for_cluster = 100;

    double min_value = FLT_MAX;
    double max_value = -FLT_MAX;
    double a = 1.0;

    grid_uniform(&g, v3d_c(0, 0, -1));

    grid_create_skyrmion_at(&g, 4, 3, 40, 60, 1, 1, M_PI);
    grid_create_skyrmion_at(&g, 4, 3, 40, 45, 1, 1, M_PI);
    grid_create_skyrmion_at(&g, 4, 3, 40, 30, 1, 1, M_PI);
    grid_create_skyrmion_at(&g, 4, 3, 40, 15, 1, 1, M_PI);

    grid_create_skyrmion_at(&g, 4, 3, 35, 60, 1, 1, M_PI);
    grid_create_skyrmion_at(&g, 4, 3, 35, 45, 1, 1, M_PI);
    grid_create_skyrmion_at(&g, 4, 3, 35, 30, 1, 1, M_PI);
    grid_create_skyrmion_at(&g, 4, 3, 35, 15, 1, 1, M_PI);

    grid_create_skyrmion_at(&g, 4, 3, 20, 60, 1, 1, M_PI);
    grid_create_skyrmion_at(&g, 4, 3, 20, 45, 1, 1, M_PI);
    grid_create_skyrmion_at(&g, 4, 3, 20, 30, 1, 1, M_PI);
    grid_create_skyrmion_at(&g, 4, 3, 20, 15, 1, 1, M_PI);

    grid_create_skyrmion_at(&g, 4, 2, 5, 60, 1, 1, M_PI);
    grid_create_skyrmion_at(&g, 4, 2, 5, 45, 1, 1, M_PI);
    grid_create_skyrmion_at(&g, 4, 2, 5, 30, 1, 1, M_PI);
    grid_create_skyrmion_at(&g, 4, 2, 5, 15, 1, 1, M_PI);

    for (double x = 0; x <= 1.0; x += 1.0 / cols) {
        double value = sin(2.0 * M_PI * x * a) + 0.25 * sin(4.0 * M_PI * x * a);
        min_value = value < min_value? value: min_value;
        max_value = value > max_value? value: max_value;
    }
    
    for (uint64_t i = 0; i < rows; ++i) {
        for (uint64_t j = 0; j < cols; ++j) {
            double x = j / (double)cols;
            double value = sin(2.0 * M_PI * x * a) + 0.25 * sin(4.0 * M_PI * x * a);
            value = (value - min_value) / (max_value - min_value);
            grid_set_anisotropy_loc(&g, i, j, anisotropy_z_axis(value * 0.03 * J + 0.05 * J));
        }
    }

    double f = 16.7e9;
    double H = 0.5 * dm * dm / J * 1.0 / mu;
    double angle = M_PI / 4.0;

    int_params.field_func = str_from_fmt("return v3d_c(%e * sin(2.0 * M_PI * %e * time + 250.0 * M_PI / 180.0) * sin(%e), 0, 0 * %e * sin(2.0 * M_PI * %e * time) * cos(%e) + %e);\n", H, f, M_PI_2, H, f, 0, H);

    grid_renderer_integrate(&g, int_params, 500, 500);

    str_free(&int_params.field_func);
    grid_free(&g);
    return 0;
}

int ratchet_testing2(void) {
    unsigned int rows = 128;
    unsigned int cols = 128;

    double lattice = 0.5e-9;
    double J = 50 * KB;
    double dm = 0.15 * J;
    double alpha = 0.1;

    grid g = grid_init(rows, cols);

    double mu = 6.74 * MU_B;

    grid_set_lattice(&g, lattice);
    grid_set_alpha(&g, alpha);
    grid_set_exchange(&g, J);
    grid_set_mu(&g, mu);
    grid_set_anisotropy(&g, anisotropy_z_axis(0));

    dm_interaction default_dm = dm_bulk(dm);
    grid_set_dm(&g, default_dm);


    double dt = 0.05 * HBAR / (J * SIGN(J));
    double ratio = (double)rows / cols;

    logging_log(LOG_INFO, "Integration dt: %e", dt);
    
    integrate_params int_params = integrate_params_init();
    int_params.field_func = create_field_D2_over_J(v3d_c(0, 0, 0.1), J, dm, mu);
    int_params.duration = 200.2 * NS;
    int_params.interval_for_raw_grid = 1000;
    int_params.dt = dt;
    int_params.do_cluster = true;
    int_params.interval_for_cluster = 100;

    grid_uniform(&g, v3d_c(0, 0, 1));

    grid_create_skyrmion_at(&g, 20, 10, cols / 2, rows / 2, -1, 1, -M_PI / 2.0);

    double hz = 0.01 * J / mu;
    int_params.field_func = create_field_tesla(v3d_c(0, 0, hz));
    grid_renderer_integrate(&g, int_params, 500, 500);

    double f = 16.7e9;
    double hx = 0.003 * J / mu;
    double m = 2;
    double n = 3;
    double omega_1 = m * f;
    double omega_2 = n * f;

    int_params.field_func = str_from_fmt("return v3d_c(%e * (sin(2.0 * M_PI * %e * time) + sin(2.0 * M_PI * %e * time)), 0, %e);\n", hx, omega_1, omega_2, hz);

    grid_renderer_integrate(&g, int_params, 500, 500);

    str_free(&int_params.field_func);
    grid_free(&g);
    return 0;
}

int testing(void) {
    unsigned int rows = 64;
    unsigned int cols = 64;

    double lattice = 0.5e-9;
    double J = 1.0e-3 * QE;
    double dm = 0.5 * J;
    double ani = 0.05 * J;
    double alpha = 0.3;

    grid g = grid_init(rows, cols);

    double mu = g.gp->mu;
    grid_set_lattice(&g, lattice);
    grid_set_alpha(&g, alpha);
    grid_set_exchange(&g, J);
    grid_set_anisotropy(&g, (anisotropy){.dir=v3d_c(0.0, 0.0, 1.0), .ani = ani});
    grid_set_mu(&g, mu);
    grid_fill_with_random(&g);

    dm_interaction default_dm = dm_interfacial(-dm);
    grid_set_dm(&g, default_dm);

    double dt = 0.01 * HBAR / (J * SIGN(J));
    double ratio = (double)rows / cols;

    logging_log(LOG_INFO, "Integration dt: %e", dt);
    
    integrate_params int_params = integrate_params_init();

    int_params.field_func = create_field_D2_over_J(v3d_c(0, 0, 0.3), J, dm, mu);
    int_params.duration = 200.2 * NS;
    int_params.interval_for_raw_grid = 1000;
    int_params.dt = dt;
    int_params.do_cluster = true;
    int_params.interval_for_cluster = 100;

    grid_renderer_integrate(&g, int_params, 1000, 1000);

    int_params.temperature_func = create_temperature(20);
    grid_renderer_integrate(&g, int_params, 1000, 1000);

    grid_free(&g);
    return 0;
}

int testing3(void) {
    grid g = {0};
    if (!grid_from_file(str_is_cstr("/home/jose/dados/dimer/dynamics/dimer/input.bin"), &g))
        logging_log(LOG_FATAL, "a");

    int rows = g.gi.rows;
    int cols = g.gi.cols;
    double lattice = g.gp->lattice;
    double J = g.gp->exchange;
    double dm = sqrt(v3d_dot(g.gp->dm.dmv_up, g.gp->dm.dmv_up));
    double alpha = g.gp->alpha;
    double mu = g.gp->mu;

    double dt = 0.01 * HBAR / (J * SIGN(J));
    double ratio = (double)rows / cols;
    
    integrate_params int_params = integrate_params_init();
    int_params.field_func = create_field_D2_over_J(v3d_c(0, 0, -0.6), J, dm, mu);
    int_params.duration = 200.2 * NS;
    int_params.interval_for_raw_grid = 1000;
    int_params.dt = dt;
    int_params.do_cluster = true;
    int_params.interval_for_cluster = 100;

    gpu_optimal_wg = 64;
    grid_renderer_integrate(&g, int_params, 1000, 1000);

    int_params.temperature_func = create_temperature(0.1);
    grid_renderer_integrate(&g, int_params, 1000, 1000);

    grid_free(&g);
    return 0;
}

int main(void) {
    testing3();
    //ratchet_testing2();
    //skyrmionium_testing();
    //testing();
    return 0;
}

