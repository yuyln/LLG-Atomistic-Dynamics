#include "atomistic_simulation.h"

int main(void) {
    grid g = grid_init(130, 130);
    int rows = g.gi.rows;
    int cols = g.gi.cols;
    double J = 1.0e-3 * QE;
    double dm = 0.2 * J;
    double ani = 0.02 * J;
    double mu = g.gp->mu;
    grid_set_exchange(&g, J);
    grid_set_dm(&g, dm_interfacial(dm));
    grid_set_anisotropy(&g, anisotropy_z_axis(ani));
    grid_set_mu(&g, mu);

    gradient_descent_params gd = gradient_descent_params_init();
    gd.T = 500;
    gd.steps = 300000;
    gd.outer_steps = 10;
    gd.damping = 0;
    gd.T_factor = pow(1.0e-5 / gd.T, 1.0 / gd.steps);
    gd.field_func = create_field_D2_over_J(v3d_c(0, 0, 0.5), J, dm, mu);

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            if ((i - rows / 2) * (i - rows / 2) + (j - cols / 2) * (j - cols / 2) >= 64 * 64)
                g.gp[i * cols + j].pin = (pinning){.dir = v3d_c(0, 0, 1), .pinned = 1};

    grid_renderer_gradient_descent(&g, gd, 1000, 1000);

    bool ret = grid_dump_path("lattice.bin", &g);
    grid_free(&g);
    return 0;
}

