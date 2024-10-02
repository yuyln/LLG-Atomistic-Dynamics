#include "atomistic_simulation.h"

void apply_current(double cur, int n_defects) {
    grid g = {0};
    if (!grid_from_animation_bin("./input.bin", &g, -1))
        logging_log(LOG_FATAL, "Could not open file");

    double J = g.gp->exchange;
    double dm = sqrt(v3d_dot(g.gp->dm.dmv_up, g.gp->dm.dmv_up));
    double mu = g.gp->mu;

    const char *base_path = str_fmt_tmp("./data/%.5e_%d", cur, n_defects);
    const char *command = str_fmt_tmp("mkdir -p %s", base_path);
    system(command);

    integrate_params int_params = integrate_params_init();
    int_params.field_func = create_field_D2_over_J(v3d_c(0, 0, 0.5), J, dm, mu);
    int_params.dt = 0.05 * HBAR / fabs(J);
    int_params.output_path = base_path;
    int_params.interval_for_information = 1000;
    int_params.interval_for_raw_grid = 0;
    int_params.interval_for_rgb_grid = 50000;
    int_params.current_func = create_current_she_dc(cur, v3d_c(0, 1, 0), 1);
    int_params.duration = 200 * NS;

    srand(111);
    uint64_t count = 0;
    for (int i = 0; i < n_defects; ++i) {
        int x = shit_random(0, 1) * g.gi.cols;
        int y = shit_random(0, 1) * g.gi.rows;
        if ((x - g.gi.cols / 2) * (x - g.gi.cols / 2) + (y - g.gi.rows / 2) * (y - g.gi.rows / 2) <= (25 * 25))
            continue;
        g.gp[y * g.gi.cols + x].ani.ani = 0.05 * J;
        count += 1;
    }
    logging_log(LOG_INFO, "%lu defects included = %e%%", count, (double)count * 100.0 / (g.gi.rows * g.gi.cols));
    grid_renderer_integrate(&g, int_params, 1000, 1000);

    grid_free(&g);
}

int main(void) {
    p_id = 1;
    grid g = grid_init(272, 272);

    double J = g.gp->exchange;
    double dm = 0.5 * J;
    double mu = g.gp->mu;

    grid_set_mu(&g, mu);
    grid_set_exchange(&g, J);
    grid_set_dm(&g, dm_interfacial(dm));
    grid_set_anisotropy(&g, anisotropy_z_axis(0.05 * J));

    grid_uniform(&g, v3d_c(0, 0, 1));
    for (uint64_t i = g.gi.cols / 2; i < g.gi.cols; ++i) {
        for (uint64_t j = 0; j < g.gi.rows; ++j) {
            double d = (i - g.gi.cols / 2.0) * (i - g.gi.cols / 2.0) + (j - g.gi.rows / 2.0) * (j - g.gi.rows / 2.0);
            if (d >= 20 * 20 && d <= 25 * 25) {
                g.m[j * g.gi.cols + i] = v3d_c(0, 0, -1);
            }
        }
    }

    grid_create_skyrmion_at(&g, 10, 5, g.gi.cols / 2, g.gi.rows / 2, -1, 1, M_PI);
    grid_create_skyrmion_at(&g, 10, 5, g.gi.cols / 2 - 20, g.gi.rows / 2, -1, 1, M_PI);

    integrate_params ip = integrate_params_init();
    ip.field_func = create_field_D2_over_J(v3d_c(0, 0, 0.5), J, dm, mu);

    grid_renderer_integrate(&g, ip, 1000, 1000);

    ip.current_func = create_current_stt_dc(0, 10e10, 0);
    grid_renderer_integrate(&g, ip, 1000, 1000);
    return 0;
}
