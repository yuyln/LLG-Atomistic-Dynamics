#include "atomistic_simulation.h"
#include "grid_funcs.h"
#include <stdint.h>
#include <float.h>


int main(void) {
    grid g = grid_init(32, 32, 16);
    double J = 1.0e-3 * QE;
    double dm = 0.2 * J;
    double ani = 0.02 * J;
    
    //double dm = -0.8 * J;
    //double ani = 0.5 * J;

    grid_set_dm(&g, dm_bulk(dm));
    grid_set_exchange(&g, isotropic_exchange(J));
    grid_set_anisotropy(&g, anisotropy_z_axis(ani));

    //for (uint64_t i = 0; i < g.gi.rows; ++i) {
    //    for (uint64_t j = 0; j < g.gi.cols; ++j) {
    //        grid_set_anisotropy_loc(&g, i, j, 0, anisotropy_z_axis(2 * J));
    //        grid_set_anisotropy_loc(&g, i, j, g.gi.depth - 1, anisotropy_z_axis(2 * J));
    //    }
    //}

    //for (uint64_t k = 0; k < g.gi.depth; ++k)
    //    V_AT(g.gp, g.gi.rows / 2, g.gi.cols / 2, k, g.gi.rows, g.gi.cols).pin = (pinning){.dir = v3d_c(0, 0, -1), .pinned = 1};

    integrate_params ip = integrate_params_init();
    ip.dt = 0.01 * HBAR / J;
    ip.interval_for_raw_grid = 500;
    ip.interval_for_information = 10;
    ip.field_func = create_field_D2_over_J(v3d_c(0, 0, 0.5), J, dm, g.gp->mu);
    logging_log(LOG_INFO, "%s", ip.field_func);

    gradient_descent_params gd = gradient_descent_params_init();
    gd.field_func = ip.field_func;
    gd.T = 0;
    gd.damping = 1;
    gd.dt = 0.01;
    gd.T_factor = 0.9995;

    grid_uniform(&g, v3d_c(0, 0, 1));
    grid_create_skyrmion_at(&g, 5, 5, g.gi.cols / 2, g.gi.rows / 2, -1, 1, M_PI);
    //grid_create_hopfion_at(&g, g.gi.cols / 4.0, 4, g.gi.cols / 2, g.gi.rows / 2, g.gi.depth / 2);

    //grid_renderer_gradient_descent(&g, gd, 1000, 1000);
    grid_renderer_integrate(&g, ip, 1000, 1000);
    return 0;
}
