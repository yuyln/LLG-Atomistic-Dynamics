#include "atomistic_simulation.h"
#include "grid_funcs.h"
#include <stdint.h>
#include <float.h>


int main(void) {
    grid g = grid_init(16, 16, 32);
    double J = 1.0e-3 * QE;
    double dm = 0.8 * J;
    double ani = 0.0 * J;

    g.gi.pbc.pbc_z = false;

    grid_set_dm(&g, dm_bulk(dm));
    grid_set_exchange(&g, isotropic_exchange(J));
    grid_set_anisotropy(&g, anisotropy_z_axis(ani));
    //grid_uniform(&g, v3d_normalize(v3d_c(1, 0, -1)));
    //for (uint64_t i = 0; i < g.gi.rows; ++i)
    //    for (uint64_t j = 0; j < g.gi.cols; ++j)
    //        for (uint64_t k = 0; k < g.gi.depth; ++k)
    //            V_AT(g.m, i, j, k, g.gi.rows, g.gi.cols) = v3d_normalize(v3d_c(cos(8.0 * M_PI * (double)k / g.gi.depth), sin(8.0 * M_PI * (double)k / g.gi.depth), cos(8.0 * M_PI * (double)k / g.gi.depth)));

    for (uint64_t i = 0; i < g.gi.rows; ++i) {
        for (uint64_t j = 0; j < g.gi.cols; ++j) {
            grid_set_dm_loc(&g, i, j, 0, dm_interfacial(dm));
            grid_set_dm_loc(&g, i, j, g.gi.depth - 1, dm_interfacial(dm));
            //grid_set_anisotropy_loc(&g, i, j, 0, anisotropy_z_axis(0.05 * J));
            //grid_set_anisotropy_loc(&g, i, j, g.gi.depth - 1, anisotropy_z_axis(0.05 * J));
        }
    }

    //for (uint64_t k = 0; k < g.gi.depth; ++k)
    //    V_AT(g.gp, g.gi.rows / 2, g.gi.cols / 2, k, g.gi.rows, g.gi.cols).pin = (pinning){.dir = v3d_c(0, 0, -1), .pinned = 1};

    integrate_params ip = integrate_params_init();
    ip.dt = 0.01 * HBAR / J;
    ip.interval_for_raw_grid = 500;
    ip.field_func = create_field_D2_over_J(v3d_c(0, 0.0, 0.3), J, dm, g.gp->mu);

    gradient_descent_params gd = gradient_descent_params_init();
    gd.field_func = ip.field_func;
    gd.T = 50;
    gd.damping = 1;
    gd.dt = 0.01;

    grid_renderer_gradient_descent(&g, gd, 1000, 1000);
    grid_renderer_integrate(&g, ip, 1000, 1000);
    return 0;
}
