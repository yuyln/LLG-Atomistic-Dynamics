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

/*
int circular(void) {
    grid g = grid_init(60, 60);
    int rows = g.gi.rows;
    int cols = g.gi.cols;
    double lattice = g.gp->lattice;
    double J = g.gp->exchange;
    double dm = 0.5 * J;
    double ani = 0.02 * J;
    double mu = g.gp->mu;

    grid_set_dm(&g, dm_interfacial(dm));
    grid_set_anisotropy(&g, anisotropy_z_axis(ani));
 
    double dt = 0.01 * HBAR / (J * SIGN(J));
    double ratio = (double)rows / cols;

    for (int64_t i = 0; i < g.gi.rows; ++i)
        for (int64_t j = 0; j < g.gi.cols; ++j)
            if (((i - 30) * (i - 30) + (j - 30) * (j - 30)) > (25 * 25))
                g.gp[i * g.gi.cols + j].pin = (pinning){.pinned = 1, .dir = v3d_c(0, 0, 1)};

    gradient_descent_params gd = gradient_descent_params_init();
    gd.damping = 0.1;
    gd.T = 500;
    gd.T_factor = 0.99999;
    gd.field_func = create_field_D2_over_J(v3d_c(0, 0, 0.5), J, dm, mu);
    grid_renderer_gradient_descent(&g, gd, 1000, 1000);

    grid_free(&g);

    return 0;
}
*/

int main(void) {
    return antiskyrmion();
}

