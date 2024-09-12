#include "atomistic_simulation.h"
#include "grid_funcs.h"
#include "kernel_funcs.h"
#include <stdint.h>
#include <float.h>
#include <assert.h>
#include "stb_image_write.h"

//ignore
int hopfion(void) {
    steps_per_frame = 1;
    grid g = grid_init(128, 128, 32);

    double lattice = 0.45598e-9;
    double J = exchange_from_micromagnetic(15e-12, lattice, 1);//exchange_from_micromagnetic(0.16e-12, lattice, 1);
    double dm = dm_from_micromagnetic(6e-3, lattice, 1);//dm_from_micromagnetic(0.115e-3, lattice, 1);
    double ani = anisotropy_from_micromagnetic(0.8e6, lattice, 1);//anisotropy_from_micromagnetic(40e3, lattice, 1);
    double mu = mu_from_micromagnetic(580e3, lattice, 1);//mu_from_micromagnetic(1.51e5, lattice, 1);
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

    g.gi.pbc.pbc_z = 0;

    for (uint64_t i = 0; i < g.gi.rows; ++i) {
        for (uint64_t j = 0; j < g.gi.cols; ++j) {
            V_AT(g.gp, i, j, 0, g.gi.rows, g.gi.cols).ani.ani = 2 * J;
            V_AT(g.gp, i, j, g.gi.depth - 1, g.gi.rows, g.gi.cols).ani.ani = 2 * J;
        }
    }

    grid_uniform(&g, v3d_c(0, 0, 1));
    grid_create_hopfion_at(&g, 10, 10, 0.5, g.gi.cols / 2, g.gi.rows / 2, g.gi.depth / 2);

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

int aaa(void) {
    int width = 2000;
    int height = 2000;
    double dt = 1.0 / width;
    double dl = 1.0 / height;
    RGBA32 *colors = mmalloc(sizeof(*colors) * width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int yl = height - y - 1;
            RGBA32 c = hsl_to_rgb(x * dt, 1, yl * dl);
            uint8_t tmp = c.r;
            c.r = c.b;
            c.b = tmp;
            colors[y * width + x] = c;
        }
    }
    stbi_write_png("./colormap.png", width, height, 4, colors, width * sizeof(*colors));
    return 0;
}

int main(void) {
    //return conical_skyrmion();
    //return hopfion();
    steps_per_frame = 2;
    grid g = {0};
    if (!grid_from_animation_bin("./hopfion.bin", &g, -1))
        logging_log(LOG_FATAL, "A");

    for (uint64_t i = 0; i < g.dimensions; ++i)
        g.gp[i].pin = (pinning){0};

    double J = fabs(g.gp->exchange.J_up);
    double dm = fabs(g.gp->dm.dmv_up.y);
    double mu = g.gp->mu;
    integrate_params ip = integrate_params_init();
    ip.dt = 0.05 * HBAR / J;
    ip.interval_for_raw_grid = 500;
    ip.interval_for_information = 100;
    ip.field_func = create_field_D2_over_J(v3d_c(0, 0, 0.0), J, dm, g.gp->mu);
    logging_log(LOG_INFO, "%s", ip.field_func);
    ip.current_func = create_current_stt_dc(10e10 * 0.12, 0, 1);
    grid_renderer_integrate(&g, ip, 1000, 1000);
    return 0;

    return hopfion();
#if 0
    grid g = grid_init(32, 32, 16);
    double J = 1.0e-3 * QE;
    double dm = 0.5 * J;
    double ani = 0.00 * J;
    
    grid_set_dm(&g, dm_bulk(dm));
    grid_set_exchange(&g, isotropic_exchange(J));
    grid_set_anisotropy(&g, anisotropy_z_axis(ani));
    g.gi.pbc.pbc_z = true;
    {
        dm_interaction dmi = dm_bulk(dm);
        dmi.dmv_front = v3d_s(0);
        dmi.dmv_back = v3d_s(0);
        grid_set_dm(&g, dmi);
    }

    //for (uint64_t i = 0; i < g.gi.rows; ++i) {
    //    for (uint64_t j = 0; j < g.gi.cols; ++j) {
    //        //grid_set_anisotropy_loc(&g, i, j, 0, anisotropy_z_axis(2 * J));
    //        //grid_set_anisotropy_loc(&g, i, j, g.gi.depth - 1, anisotropy_z_axis(2 * J));
    //        grid_set_dm_loc(&g, i, j, 0, dm_interfacial(dm));
    //        grid_set_dm_loc(&g, i, j, g.gi.depth - 1, dm_interfacial(dm));
    //    }
    //}

    //for (uint64_t k = 0; k < g.gi.depth; ++k)
    //    V_AT(g.gp, g.gi.rows / 2, g.gi.cols / 2, k, g.gi.rows, g.gi.cols).pin = (pinning){.dir = v3d_c(0, 0, -1), .pinned = 1};

    integrate_params ip = integrate_params_init();
    ip.dt = 0.01 * HBAR / J;
    ip.interval_for_raw_grid = 500;
    ip.interval_for_information = 100;
    ip.field_func = create_field_D2_over_J(v3d_c(0, 0, 0.5), J, dm, g.gp->mu);
    logging_log(LOG_INFO, "%s", ip.field_func);

    gradient_descent_params gd = gradient_descent_params_init();
    gd.field_func = ip.field_func;
    gd.T = 0;
    gd.damping = 1;
    gd.dt = 0.01;
    gd.T_factor = 0.9995;

    grid_uniform(&g, v3d_c(0, 0, 1));
    for (uint64_t k = 0; k < g.gi.depth; ++k)
        V_AT(g.gp, g.gi.rows / 2, g.gi.cols / 2, k, g.gi.rows, g.gi.cols).pin = (pinning){.pinned = 1, .dir=v3d_c(0, 0, -1)};

    grid_renderer_integrate(&g, ip, 1000, 1000);

    for (uint64_t k = 0; k < g.gi.depth; ++k)
        V_AT(g.gp, g.gi.rows / 2, g.gi.cols / 2, k, g.gi.rows, g.gi.cols).pin = (pinning){.pinned = 0, .dir=v3d_c(0, 0, -1)};

    //ip.current_func = str_fmt_tmp("return (current){.type = CUR_STT, .stt.j = v3d_c(%.15e * sin(2.0 * M_PI * gs.k / %e), .stt.beta = 0, .stt.polarization = -1};", 10e10, (double)g.gi.depth);
    //logging_log(LOG_INFO, ip.current_func);
    //grid_renderer_integrate(&g, ip, 1000, 1000);
    return 0;
#endif
}
