#include "atomistic_simulation.h"
#include "grid_funcs.h"
#include "kernel_funcs.h"
#include <stdint.h>
#include <float.h>
#include <assert.h>
#include "stb_image_write.h"

int bilayer(void) {
    steps_per_frame = 2;
    grid g = grid_init(128, 128, 20);

    double lattice = 0.5e-9;
    double J = 1.0e-3 * QE;
    double dm = 0.2 * J;
    double ani = 0.05 * J;
    double mu = g.gp->mu;
    logging_log(LOG_INFO, "J = %e eV", J / QE);
    logging_log(LOG_INFO, "D/J = %e", dm / J);
    logging_log(LOG_INFO, "K/J = %e", ani / J);
    logging_log(LOG_INFO, "mu = %e mu_B", mu / MU_B);
    logging_log(LOG_INFO, "mu/hbar gamma = %e", mu / (HBAR * g.gp->gamma));
    logging_log(LOG_INFO, "gamma = %e", g.gp->gamma);

    exchange_interaction Je = isotropic_exchange(J);
    Je.J_front *= -1;
    Je.J_back *= -1;
    grid_set_alpha(&g, 0.3);
    grid_set_dm(&g, dm_bulk(dm));
    grid_set_exchange(&g, Je);
    grid_set_anisotropy(&g, anisotropy_z_axis(ani));
    grid_set_mu(&g, mu);
    grid_set_lattice(&g, lattice);

    grid_uniform(&g, v3d_c(0, 0, 1));
    grid_create_skyrmion_at(&g, 15, 5, g.gi.cols / 2, g.gi.rows / 2, -1, 1, M_PI / 2);
    for (uint64_t i = 0; i < g.gi.rows; ++i)
        for (uint64_t j = 0; j < g.gi.cols; ++j)
            for (uint64_t k = 0; k < g.gi.depth; k += 2)
                V_AT(g.m, i, j, k, g.gi.rows, g.gi.cols) = v3d_scalar(V_AT(g.m, i, j, 0, g.gi.rows, g.gi.cols), -1);
    integrate_params ip = integrate_params_init();
    ip.dt = 0.05 * HBAR / J;
    ip.interval_for_raw_grid = 500;
    ip.interval_for_information = 100;
    ip.field_func = create_field_D2_over_J(v3d_c(0, 0, 0.5), J, dm, g.gp->mu);
    ip.current_func = create_current_she_dc(1e10, v3d_c(0, 1, 0), 0);
    logging_log(LOG_INFO, "%s", ip.field_func);

    gradient_descent_params gd = gradient_descent_params_init();
    gd.field_func = ip.field_func;
    gd.T = 0;
    gd.damping = 0;
    gd.dt = 0.1;
    gd.T_factor = 0.9995;
    g.gi.pbc.pbc_x = 0;
    g.gi.pbc.pbc_y = 0;
    //grid_renderer_gradient_descent(&g, gd, 1000, 1000);
    grid_renderer_integrate(&g, ip, 1000, 1000);

    g.gi.pbc.pbc_x = 1;
    g.gi.pbc.pbc_y = 1;
    grid_renderer_integrate(&g, ip, 1000, 1000);
    return 0;
}

int hopfion(void) {
    steps_per_frame = 1;
    grid g = grid_init(256, 256, 32);

    double lattice = 0.5e-9;
    //double lattice = 0.5e-9;
    double J   = 1.0e-3 * QE;//exchange_from_micromagnetic(0.16e-12, lattice, 1);
    double dm  = 0.2 * J;//dm_from_micromagnetic(0.115e-3, lattice, 1);
    double ani = 0.02 * J;//anisotropy_from_micromagnetic(40e3, lattice, 1);
    double mu  = g.gp->mu;//mu_from_micromagnetic(1.51e5, lattice, 1);
    logging_log(LOG_INFO, "J = %e eV", J / QE);
    logging_log(LOG_INFO, "D/J = %e", dm / J);
    logging_log(LOG_INFO, "K/J = %e", ani / J);
    logging_log(LOG_INFO, "mu = %e mu_B", mu / MU_B);
    logging_log(LOG_INFO, "mu/hbar gamma = %e", mu / (HBAR * g.gp->gamma));
    logging_log(LOG_INFO, "gamma = %e", g.gp->gamma);

    grid_set_alpha(&g, 0.3);
    grid_set_dm(&g, dm_bulk(dm));
    grid_set_exchange(&g, isotropic_exchange(J));
    grid_set_anisotropy(&g, anisotropy_z_axis(ani));
    grid_set_mu(&g, mu);
    grid_set_lattice(&g, lattice);

    grid_uniform(&g, v3d_c(0, 0, 1));
    grid_create_hopfion_at(&g, 20, 20, 0.5, g.gi.cols / 2, g.gi.rows / 2, g.gi.depth / 2);

    g.gi.pbc.pbc_z = 0;

    for (uint64_t i = 0; i < g.gi.rows; ++i) {
        for (uint64_t j = 0; j < g.gi.cols; ++j) {
            V_AT(g.gp, i, j, 0, g.gi.rows, g.gi.cols).ani.ani = 2 * J;
            V_AT(g.gp, i, j, g.gi.depth - 1, g.gi.rows, g.gi.cols).ani.ani = 2 * J;

            V_AT(g.m, i, j, 0, g.gi.rows, g.gi.cols) = v3d_c(0, 0, 1);
            V_AT(g.m, i, j, g.gi.depth - 1, g.gi.rows, g.gi.cols) = v3d_c(0, 0, 1);
        }
    }

    integrate_params ip = integrate_params_init();
    ip.dt = 0.05 * HBAR / J;
    ip.interval_for_raw_grid = 500;
    ip.interval_for_information = 100;
    ip.field_func = create_field_D2_over_J(v3d_c(0, 0, 0.0), J, dm, g.gp->mu);
    logging_log(LOG_INFO, "%s", ip.field_func);

    gradient_descent_params gd = gradient_descent_params_init();
    gd.field_func = ip.field_func;
    gd.T = 0;
    gd.damping = 0;
    gd.dt = 0.1;
    gd.T_factor = 0.9995;
    grid_renderer_gradient_descent(&g, gd, 1000, 1000);

    grid_renderer_integrate(&g, ip, 1000, 1000);
    return 0;
}

int conical_skyrmion(void) {
    steps_per_frame = 1;
    grid g = grid_init(64, 272, 64);
    double J = 1.0e-3 * QE;
    double dm = 0.18 * J;
    double ani = 0.00 * J;
    
    grid_set_dm(&g, dm_bulk(dm));
    grid_set_exchange(&g, isotropic_exchange(J));
    grid_set_anisotropy(&g, anisotropy_z_axis(ani));

    integrate_params ip = integrate_params_init();
    ip.dt = 0.01 * HBAR / J;
    ip.interval_for_raw_grid = 0;
    ip.interval_for_information = 0;
    ip.field_func = create_field_D2_over_J(v3d_c(0, 0, 0.5), J, dm, g.gp->mu);
    logging_log(LOG_INFO, "%s", ip.field_func);

    gradient_descent_params gd = gradient_descent_params_init();
    gd.field_func = ip.field_func;
    gd.T = 0;
    gd.damping = 1;
    gd.dt = 0.01;
    gd.T_factor = 0.9995;
    gd.outer_steps = 1;
    gd.steps = 30000;

    grid_uniform(&g, v3d_c(0, 0, 1));

    for (uint64_t i = 0; i < g.gi.rows; ++i) {
        double y = (double)i / g.gi.rows;
        for (uint64_t j = 0; j < g.gi.cols; ++j) {
            double x = (double)j / g.gi.cols;
            for (uint64_t k = 0; k < g.gi.depth; ++k) {
                double z = (double)k / g.gi.depth;
                double mx = sin(4.0 * M_PI * z);
                double my = cos(4.0 * M_PI * z);
                double mz = 0.5;
                V_AT(g.m, i, j, k, g.gi.rows, g.gi.cols) = v3d_normalize(v3d_c(mx, my, mz));
            }
        }
    }

    for (uint64_t k = 0; k < g.gi.depth; ++k) {
        V_AT(g.gp, g.gi.rows / 2, 0, k, g.gi.rows, g.gi.cols).pin = (pinning){.pinned = 1, .dir=v3d_c(0, 0, -1)};
        V_AT(g.gp, g.gi.rows / 2, g.gi.cols / 2, k, g.gi.rows, g.gi.cols).pin = (pinning){.pinned = 1, .dir=v3d_c(0, 0, -1)};
    }

    grid_create_skyrmion_at(&g, 10, 1, 0, g.gi.rows / 2, -1, 1, -M_PI / 2);
    grid_create_skyrmion_at(&g, 10, 1, g.gi.cols / 2, g.gi.rows / 2, -1, 1, -M_PI / 2);

    grid_renderer_gradient_descent(&g, gd, 1000, 1000);
    grid_dump_path("./guess.bin", &g);

    //ip.current_func = create_current_stt_dc(10e10, 0, 0);
    //grid_renderer_integrate(&g, ip, 1000, 1000);
    return 0;

}

int main(void) {
    p_id = 1;
    //return bilayer();
    //return conical_skyrmion();
    return hopfion();
    steps_per_frame = 1;
    grid g = {0};
    if (!grid_from_animation_bin("./hopfion.bin", &g, -1))
        logging_log(LOG_FATAL, "A");

    for (uint64_t i = 0; i < g.dimensions; ++i)
        g.gp[i].pin = (pinning){0};

    grid_set_alpha(&g, 0.3);

    double J = fabs(g.gp->exchange.J_up);
    double dm = fabs(g.gp->dm.dmv_up.y);
    double mu = g.gp->mu;
    logging_log(LOG_INFO, "J = %.e eV", J / QE);
    logging_log(LOG_INFO, "D = %.e J", dm / J);
    logging_log(LOG_INFO, "K = %.e J", g.gp[256 * 256 + 1].ani.ani / J);
    integrate_params ip = integrate_params_init();
    ip.dt = 0.05 * HBAR / J;
    ip.interval_for_raw_grid = 500;
    ip.interval_for_information = 100;
    ip.current_func = create_current_stt_dc(1e10 * 0.1, 0, 0.1);
    ip.current_func = create_current_she_dc(1e10 * 0.1, v3d_c(1, 0, 0), 0);
    grid_renderer_integrate(&g, ip, 1000, 1000);
    return 0;
}
