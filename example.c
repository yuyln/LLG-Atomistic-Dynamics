#include "atomistic_simulation.h"

void example(grid *g, unsigned long row, unsigned long col, void *user_data) {
    double ani = *(double*)user_data;
    //grid_set_anisotropy_loc(g, row, col, anisotropy_z_axis(ani));
    grid_set_pinning_loc(g, row, col, (pinning){.pinned = 1, .dir = v3d_c(0, 0, -1)});
}

int main(void) {
    int rows = 64;
    int cols = 64;
    grid g = grid_init(rows, cols);

    double lattice = 0.5e-9;
    //double alpha = 0.04;
    double alpha = 0.4;
    double J = 1.0e-3 * QE;
    double dm = 0.5 * J;
    double ani = 0.5 * J;
    double mu = g.gp->mu;

    grid_set_exchange(&g, J);
    grid_set_anisotropy(&g, (anisotropy){.ani = -ani, .dir = v3d_c(0, 0, 1)});
    grid_set_dm(&g, dm_interfacial(dm));
    double ani_defect = 2 * ani;

    //grid_uniform(&g, v3d_c(0, 0, 1));
    //grid_create_skyrmion_at(&g, 20, 1, cols / 2, rows / 2, -1, 1, M_PI / 2);
    //grid_create_skyrmion_at(&g, 10, 1, cols / 2, rows / 2, 1, 1, M_PI / 2);

    //int n = 6;
    //double Ry = 2.0 * rows / (3.0 * n);
    //double Rx = 2.0 * cols / (3.0 * n);

    //for (unsigned int iy = 0; iy < n; ++iy) {
    //    int yc = Ry / 4.0 + iy * (Ry + Ry / 2.0) + Ry / 2.0;
    //    for (unsigned int ix = 0; ix < n; ++ix) {
    //        int xc = Rx / 4.0 + ix * (Rx + Rx / 2.0) + Rx / 2.0;
    //        if (iy % 2 == 0)
    //            xc += Rx / 2.0 + Rx / 4.0;
    //        grid_create_skyrmion_at(&g, Rx / 2.0, 1, xc, yc, -1, 1, 0);
    //    }
    //}

    integrate_params int_params = integrate_params_init();
    int_params.dt = HBAR / J * 0.01;
    int_params.field_func = create_field_D2_over_J(v3d_c(0, 0, 0.5), J, dm, mu);
    int_params.current_func = create_current_stt_dc(5e10, 0, 0);
    grid_renderer_integrate(&g, int_params, 1000, 1000);


    grid_free(&g);

    return 0;
}
