#include "atomistic_simulation.h"
#include "render.h"

void apply_current(double cur, int n_defects) {
    grid g = {0};
    if (!grid_from_animation_bin("./input.bin", &g, -1))
        logging_log(LOG_FATAL, "Could not open file");

    double J = g.gp->exchange;
    double dm = sqrt(v3d_dot(g.gp->dm.dmv_up, g.gp->dm.dmv_up));
    double mu = g.gp->mu;

    const char *base_path = str_fmt_tmp("./data/%.5e", cur);
    const char *command = str_fmt_tmp("mkdir -p %s", base_path);
    system(command);

    integrate_params int_params = integrate_params_init();
    int_params.field_func = create_field_D2_over_J(v3d_c(0, 0, 0.5), J, dm, mu);
    int_params.dt = 0.01 * HBAR / fabs(J);
    int_params.output_path = base_path;
    int_params.interval_for_information = 1000;
    int_params.interval_for_raw_grid = 0;
    int_params.interval_for_rgb_grid = 50000;
    int_params.current_func = create_current_stt_dc(cur, 0, 1);
    int_params.duration = 200 * NS;

    srand(111);
    for (int i = 0; i < n_defects; ++i) {
        int x = shit_random(0, 1) * g.gi.cols;
        int y = shit_random(0, 1) * g.gi.rows;
        g.gp[y * g.gi.cols + x].ani.ani = 0.05 * J;
    }
    grid_renderer_integrate(&g, int_params, 1000, 1000);

    grid_free(&g);
}

int main(void) {
    double currents[] = {0.05e10, 0.075e10, 0.1e10, 0.35e10, 0.5e10};
    int currents_size = sizeof(currents) / sizeof(currents[0]);
    for (int i = 0; i < currents_size; ++i)
        apply_current(currents[i]);
    return 0;
}
