#include "atomistic_simulation.h"
#include "grid_funcs.h"
#include <stdint.h>
#include <float.h>

//@TODO: PROPER ERROR CHECKING URGENT!!!
//@TODO: check time measuring when compiling it to windows
/*int Stosic(void) {
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
    grid_renderer_integrate(&g, int_params, 1000, 1000);

    double jx = 1e10;
    double jy = 1e10;
    int_params.current_func = create_current_she_ac(jx, v3d_c(jx, jy, 0), 100 / (200 * NS), 0);
    grid_renderer_integrate(&g, int_params, 1000, 1000);

    grid_free(&g);
    return 0;
}

int ratchet_testing(void) {
    unsigned int rows = 128;
    unsigned int cols = 128;

    double lattice = 0.5e-9;
    double J = 1.0e-3 * QE;
    double dm = 0.2 * J;
    double alpha = 0.3;

    grid g = grid_init(rows, cols);

    double mu = g.gp->mu;
    grid_set_lattice(&g, lattice);
    grid_set_alpha(&g, alpha);
    grid_set_exchange(&g, J);
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
    double a = 2.0;

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
            grid_set_anisotropy_loc(&g, i, j, anisotropy_z_axis(value * 0.05 * J + 0.03 * J));
        }
    }
    {
        grid_uniform(&g, v3d_c(0, 0, -1));
        unsigned int n = 6;

        double Ry = 2.0 * rows / (3.0 * n);
        double Rx = 2.0 * cols / (3.0 * n);
        for (unsigned int iy = 0; iy < n; ++iy) {
            int yc = Ry / 4.0 + iy * (Ry + Ry / 2.0) + Ry / 2.0;
            for (unsigned int ix = 0; ix < n; ++ix) {
                int xc = Rx / 4.0 + ix * (Rx + Rx / 2.0) + Rx / 2.0;
                if (iy % 2 == 0)
                    xc += Rx / 2.0 + Rx / 4.0;
                grid_create_skyrmion_at(&g, Rx / 2, Rx / 10, xc, yc, 1, 1, M_PI);
            }
        }
        grid_renderer_integrate(&g, int_params, 500, 500);
    }

    double f = 8e9;
    double H = 0.8 * dm * dm / J * 1.0 / mu;
    double angle = M_PI / 4.0;

    int_params.field_func = str_fmt_tmp("return v3d_c(%e * sin(2.0 * M_PI * %e * time + 250.0 * M_PI / 180.0) * sin(%e), 0, 0 * %e * sin(2.0 * M_PI * %e * time) * cos(%e) + %e);\n", H, f, M_PI_4, H, f, M_PI_4, H);

    int_params.interval_for_rgb_grid = 1000;
    grid_renderer_integrate(&g, int_params, 500, 500);

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

    int_params.field_func = str_fmt_tmp("return v3d_c(%e * (sin(2.0 * M_PI * %e * time) + sin(2.0 * M_PI * %e * time)), 0, %e);\n", hx, omega_1, omega_2, hz);

    grid_renderer_integrate(&g, int_params, 500, 500);

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
    uint64_t s = 272;
    double H = 0.4;
    const char *phase = "dimer";
    if (!grid_from_animation_bin(str_fmt_tmp("/home/jose/dados/dimer/static/%s_field_vs_sample_size_6x6_triangular/data/%.5f/%llu/integrate_evolution.dat", phase, H, s), &g, -1)) {
        logging_log(LOG_FATAL, "a");
    }

    int rows = g.gi.rows;
    int cols = g.gi.cols;
    double lattice = g.gp->lattice;
    double J = g.gp->exchange;
    double dm = sqrt(v3d_dot(g.gp->dm.dmv_up, g.gp->dm.dmv_up));
    double alpha = g.gp->alpha;
    double mu = g.gp->mu;

    double min_k = FLT_MAX, max_k = -FLT_MAX;
    for (uint64_t i = 0; i < rows * cols; ++i) {
        double k = g.gp[i].ani.ani;
        max_k = k > max_k? k: max_k;
        min_k = k < min_k? k: min_k;
    }

    for (uint64_t i = 0; i < rows * cols; ++i) {
        double k = g.gp[i].ani.ani;
        k = (k - min_k) / (max_k - min_k) * 0.1 * J;
        g.gp[i].ani.ani = k;
    }

    double dt = 0.05 * HBAR / (J * SIGN(J));
    double ratio = (double)rows / cols;
    
    integrate_params int_params = integrate_params_init();
    int_params.field_func = create_field_D2_over_J(v3d_c(0, 0, -0.4), J, dm, mu);
    int_params.duration = 200.2 * NS;
    int_params.interval_for_raw_grid = 1000;
    int_params.dt = dt;
    int_params.do_cluster = true;
    int_params.interval_for_cluster = 100;

    grid_renderer_integrate(&g, int_params, 1000, 1000);

    int_params.temperature_func = create_temperature(1);
    steps_per_frame = 1;
    grid_renderer_integrate(&g, int_params, 1000, 1000);

    grid_free(&g);
    return 0;
}

void apply_circular_defect(grid *g, uint64_t row, uint64_t col, void *user_data) {
    int rows = g->gi.rows;
    int cols = g->gi.cols;
    int y = row;
    int x = col;
    g->m[y * cols + x] = v3d_c(0, 0, 1);
    g->gp[y * cols + x].exchange *= 2.0;
}

int circular_defect(void) {
    grid g = grid_init(128, 128);

    int rows = g.gi.rows;
    int cols = g.gi.cols;
    double lattice = g.gp->lattice;
    double J = g.gp->exchange;
    double dm = 0.2 * J;
    grid_set_dm(&g, dm_interfacial(dm));
    double alpha = g.gp->alpha;
    double mu = g.gp->mu;
    grid_set_anisotropy(&g, anisotropy_z_axis(0.02 * J));

    double dt = 0.01 * HBAR / (J * SIGN(J));
    double ratio = (double)rows / cols;
    
    integrate_params int_params = integrate_params_init();
    int_params.field_func = create_field_D2_over_J(v3d_c(0, 0, 0.5), J, dm, mu);
    int_params.duration = 200.2 * NS;
    int_params.interval_for_raw_grid = 1000;
    int_params.dt = dt;
    int_params.do_cluster = true;
    int_params.interval_for_cluster = 100;
    grid_uniform(&g, v3d_c(0, 0, 1));

    unsigned int n = 6;

    double Ry = 2.0 * rows / (3.0 * n);
    double Rx = 2.0 * cols / (3.0 * n);
    for (unsigned int iy = 0; iy < n; ++iy) {
        int yc = Ry / 4.0 + iy * (Ry + Ry / 2.0) + Ry / 2.0;
        for (unsigned int ix = 0; ix < n; ++ix) {
            int xc = Rx / 4.0 + ix * (Rx + Rx / 2.0) + Rx / 2.0;
            if (iy % 2 == 0)
                xc += Rx / 2.0 + Rx / 4.0;
            grid_create_skyrmion_at(&g, Rx / 2, Rx / 10, xc, yc, -1, 1, M_PI);
        }
    }

    grid_do_in_ellipse(&g, cols / 2.0, rows / 2.0, cols / 4.0, cols / 4.0, apply_circular_defect, NULL);

    gpu_optimal_wg = 32;
    grid_renderer_integrate(&g, int_params, 1000, 1000);
    int_params.current_func = create_current_stt_ac(2e10, 2e10, 1.0 / 20e-9, 0);
    grid_renderer_integrate(&g, int_params, 1000, 1000);
    //organize_clusters("./clusters.dat", "./clusters_org.dat", cols * g.gp->lattice, rows * g.gp->lattice, 1e8);
    grid_free(&g);
    return 0;
}

int Tc(void) {
    grid g = grid_init(64, 64);

    int rows = g.gi.rows;
    int cols = g.gi.cols;
    double lattice = g.gp->lattice;
    double J = g.gp->exchange * 50;
    double dm = 0.0 * J;
    grid_set_dm(&g, dm_interfacial(dm));
    double alpha = g.gp->alpha;
    double mu = 1 * g.gp->mu;
    grid_set_anisotropy(&g, anisotropy_z_axis(0.00 * J));
    grid_set_mu(&g, mu);
    grid_set_exchange(&g, J);

    double dt = 0.01 * HBAR / (J * SIGN(J));
    double ratio = (double)rows / cols;
    
    integrate_params int_params = integrate_params_init();
    //int_params.field_func = create_field_D2_over_J(v3d_c(0, 0, 0.5), J, dm, mu);
    int_params.field_func = create_field_J(v3d_c(0, 0, 0.002), J, mu);
    int_params.duration = 2 * NS;
    int_params.interval_for_raw_grid = 1000;
    int_params.dt = dt;
    int_params.do_cluster = true;
    int_params.interval_for_cluster = 100;
    struct testin {
        double mz, T;
    };
    struct {struct testin *items; uint64_t len, cap} values = {0};
    for (double T = 0; T < 100; T += 10) {
        int_params.temperature_func = create_temperature(T);
        grid_renderer_integrate(&g, int_params, 1000, 1000);
        double avg_mz = 0;
        for (uint64_t i = 0; i < rows * cols; ++i) 
            avg_mz += g.m[i].z;
        avg_mz /= rows * cols;
        da_append(&values, ((struct testin){.T = T, .mz = avg_mz}));
    }
    for (uint64_t i = 0; i < values.len; ++i)
        printf("%e,%e\n", values.items[i].T, values.items[i].mz);
    grid_free(&g);
    return 0;
}
*/

int main(void) {
    grid g = grid_init(272, 272, 32);
    //grid_uniform(&g, v3d_c(0, 0, 1));
    //grid_create_skyrmion_at(&g, 5, 3, g.gi.cols / 2, g.gi.rows / 2, -1, 1, M_PI / 2);
    integrate_params ip = integrate_params_init();
    ip.dt = 0.01 * HBAR / (1.0e-3 * QE);
    ip.duration = 200 * NS;
    ip.interval_for_raw_grid = 30000;
    g.gi.pbc.pbc_z = false;
    double dm = 0.8 * 1.0e-3 * QE;
    grid_set_dm(&g, dm_bulk(dm));
    for (uint64_t i = 0; i < g.gi.rows; ++i) {
        for (uint64_t j = 0; j < g.gi.cols; ++j) {
            grid_set_dm_loc(&g, i, j, 0, dm_interfacial(dm));
            grid_set_dm_loc(&g, i, j, g.gi.depth - 1, dm_interfacial(dm));
        }
    }

    ip.field_func = create_field_D2_over_J(v3d_c(0, 0.0, 0.6), 1e-3 * QE, dm, g.gp->mu);
    for (uint64_t k = 0; k < g.gi.depth; ++k)
        V_AT(g.gp, g.gi.rows / 2, g.gi.cols / 2, k, g.gi.rows, g.gi.cols).pin = (pinning){.dir = v3d_c(0, 0, -1), .pinned = 1};
    grid_renderer_integrate(&g, ip, 1000, 1000);
    //Tc();
    //circular_defect();
    //organize_clusters("./clusters.dat", "./clusters_org.dat", 128 * 0.5e-9, 128 * 0.5e-9, 1e8);
    //testing3();
    //ratchet_testing();
    //skyrmionium_testing();
    //testing();
    return 0;
}
