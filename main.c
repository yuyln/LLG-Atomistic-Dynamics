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
    integrate(&g, int_params);

    grid_free(&g);
}

int main(void) {
    double currents[] = {0.05e10, 0.075e10, 0.1e10, 0.35e10, 0.5e10};
    int currents_size = sizeof(currents) / sizeof(currents[0]);

    int defects[] = {10, 100, 1000, 10000, 0};
    int defects_size = sizeof(defects) / sizeof(defects[0]);

    for (int i = 0; i < currents_size; ++i)
        for (int j = 0; j < defects_size; ++j)
            apply_current(currents[i], defects[j]);
    return 0;
}
