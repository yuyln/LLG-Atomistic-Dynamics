#include "atomistic_simulation.h"

int antiskyrmion(void) {
    grid g = grid_init(64, 64);
    int rows = g.gi.rows;
    int cols = g.gi.cols;
    double lattice = g.gp->lattice;
    double J = g.gp->exchange;
    double dm = 0.1796875 * J;

    dm_interaction dm_ = dm_interfacial(dm);
    dm_.dmv_up = v3d_scalar(dm_.dmv_up, -1);
    dm_.dmv_down = v3d_scalar(dm_.dmv_down, -1);
    grid_set_dm(&g, dm_);

    double alpha = g.gp->alpha;
    double mu = g.gp->mu;
    grid_set_anisotropy(&g, anisotropy_z_axis(0.015625 * J));
    grid_set_alpha(&g, 0.3);

    double dt = 0.01 * HBAR / (J * SIGN(J));
    double ratio = (double)rows / cols;

    grid_uniform(&g, v3d_c(0, 0, 1));

    grid_create_skyrmion_at(&g, 20, 10, g.gi.cols / 2, g.gi.rows / 2, -1, -1, M_PI);

    integrate_params int_params = integrate_params_init();
    int_params.field_func = create_field_D2_over_J(v3d_c(0, 0, 0.3), J, dm, mu);
    int_params.dt = dt;
    int_params.do_cluster = true;
    int_params.interval_for_information = 1000;
    int_params.interval_for_raw_grid = 100;
    int_params.interval_for_rgb_grid = 50000;

    int_params.duration = 10 * NS;
    grid_renderer_integrate(&g, int_params, 1000, 1000);

    grid_set_alpha(&g, 0.05);
    int_params.duration = 200 * NS;
    int_params.current_func = create_current_stt_dc(10e10, 0e10, 0);
    //int_params.current_func = create_current_she_dc(2e10, v3d_c(1, 0, 0), 0);
    grid_renderer_integrate(&g, int_params, 1000, 1000);

    grid_free(&g);

    return 0;
}

int skyrminonium(void) {
    int rows = 128;
    int cols = 128;
    grid g = grid_init(rows, cols);
    double J = 1.0e-3 * QE;
    double dm = 0.2 * J;
    double ani = 0.02 * J;
    double mu = g.gp->mu;
    grid_set_exchange(&g, J);
    grid_set_dm(&g, dm_bulk(dm));
    grid_set_anisotropy(&g, anisotropy_z_axis(ani));

    grid_uniform(&g, v3d_c(0, 0, 1));
    grid_create_target_skyrmion_at(&g, 20, 10, cols / 2, rows / 2, -1, 1, -M_PI / 2.0, 2);

    integrate_params ip = integrate_params_init();
    ip.field_func = create_field_D2_over_J(v3d_c(0, 0, 0.3), J, dm, mu);
    ip.current_func = create_current_stt_dc(20e10, 0, 0);

    gradient_descent_params gd = gradient_descent_params_init();
    gd.field_func = ip.field_func;
    gd.T = 0;
    gd.damping = 1;

    grid_renderer_gradient_descent(&g, gd, 1000, 1000);
    grid_renderer_integrate(&g, ip, 1000, 1000);
    grid_free(&g);
    return 0;
}

int target(void) {
    int rows = 272;
    int cols = 272;
    grid g = grid_init(rows, cols);
    double J = 1.0e-3 * QE;
    double dm = 0.2 * J;
    double ani = 0.02 * J;
    double mu = g.gp->mu;
    grid_set_exchange(&g, J);
    grid_set_dm(&g, dm_bulk(dm));
    grid_set_anisotropy(&g, anisotropy_z_axis(ani));

    grid_uniform(&g, v3d_c(0, 0, 1));
    grid_create_target_skyrmion_at(&g, 80, 20, cols / 2, rows / 2, -1, 1, -M_PI / 2.0, 9);

    integrate_params ip = integrate_params_init();
    ip.field_func = create_field_D2_over_J(v3d_c(0, 0, 0.3), J, dm, mu);
    ip.current_func = create_current_stt_dc(20e10, 0, 0);

    gradient_descent_params gd = gradient_descent_params_init();
    gd.field_func = ip.field_func;
    gd.T = 0;
    gd.damping = 1;

    grid_renderer_gradient_descent(&g, gd, 1000, 1000);
    grid_renderer_integrate(&g, ip, 1000, 1000);
    grid_free(&g);
    return 0;
}

int temp(void) {
    int rows = 64;
    int cols = 64;
    grid g = grid_init(rows, cols);
    double lattice = 0.4689e-9;
    double J = exchange_from_micromagnetic(8.78e-12, lattice, 1);
    double dm = dm_from_micromagnetic(2e-3, lattice, 1);
    double ani = anisotropy_from_micromagnetic(800e3, lattice, 1);
    double mu = mu_from_micromagnetic(385e3, lattice, 1);
    grid_set_exchange(&g, J);
    grid_set_dm(&g, dm_bulk(dm));
    grid_set_anisotropy(&g, anisotropy_z_axis(ani));
    grid_set_mu(&g, mu);
    grid_set_lattice(&g, lattice);

    logging_log(LOG_INFO, "J = %e eV", J / QE);
    logging_log(LOG_INFO, "D/J = %e", dm / J);
    logging_log(LOG_INFO, "K/J = %e", ani / J);
    logging_log(LOG_INFO, "mu = %e mu_B", mu / MU_B);
    logging_log(LOG_INFO, "mu/hbar gamma = %e", mu / (HBAR * g.gp->gamma));
    logging_log(LOG_INFO, "gamma = %e", g.gp->gamma);

    grid_uniform(&g, v3d_c(0, 0, 1));
    grid_create_skyrmion_at(&g, 10, 5, cols / 2, rows / 2, -1, 1, -M_PI / 2.0);

    integrate_params ip = integrate_params_init();
    //ip.field_func = create_field_D2_over_J(v3d_c(0, 0, 0.5), J, dm, mu);
    ip.field_func = create_field_tesla(v3d_c(0, 0, 0.32));
    ip.temperature_func = create_temperature(100);

    gradient_descent_params gd = gradient_descent_params_init();
    gd.field_func = ip.field_func;
    gd.T = 0;
    gd.damping = 1;

    grid_renderer_gradient_descent(&g, gd, 1000, 1000);
    grid_renderer_integrate(&g, ip, 1000, 1000);
    grid_free(&g);
    return 0;
}

int test_ddi(void) {
    int rows = 64;
    int cols = 64;
    grid g = grid_init(rows, cols);
    double lattice = 0.5e-9;
    double J = 1.0e-3 * QE;
    double dm = 0.2 * J;
    double ani = 0.02 * J;
    double mu = g.gp->mu;
    grid_set_exchange(&g, J);
    grid_set_dm(&g, dm_bulk(dm));
    grid_set_anisotropy(&g, anisotropy_z_axis(ani));
    grid_set_mu(&g, mu);
    grid_set_lattice(&g, lattice);

    grid_uniform(&g, v3d_c(0, 0, 1));
    grid_create_skyrmion_at(&g, 10, 5, cols / 2, rows / 2, -1, 1, -M_PI / 2.0);

    integrate_params ip = integrate_params_init();
    ip.field_func = create_field_D2_over_J(v3d_c(0, 0, 0.5), J, dm, mu);
    ip.interval_for_information = 1;
    ip.compile_augment = "-DINCLUDE_DIPOLAR";

    gradient_descent_params gd = gradient_descent_params_init();
    gd.field_func = ip.field_func;
    gd.T = 0;
    gd.damping = 1;

    grid_renderer_gradient_descent(&g, gd, 1000, 1000);
    grid_renderer_integrate(&g, ip, 1000, 1000);
    grid_free(&g);
    return 0;

}

int main(void) {
    return temp();
    return target();
    return skyrminonium();
    return antiskyrmion();
}

