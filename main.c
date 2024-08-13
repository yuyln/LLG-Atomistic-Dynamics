#include "atomistic_simulation.h"

void apply_circular_defect(grid *g, uint64_t row, uint64_t col, void *user_data) {
    int rows = g->gi.rows;
    int cols = g->gi.cols;
    int y = row;
    int x = col;
    g->m[y * cols + x] = v3d_c(0, 0, 1);
    g->gp[y * cols + x].exchange *= 2.0;
}

int circular_defect(int radius) {
    grid g = grid_init(272, 272);

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

    string base_path = str_from_fmt("./data/%d", radius);
    string command = str_from_fmt("mkdir -p %s", str_as_cstr(&base_path));
    system(str_as_cstr(&command));
    str_free(&command);

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

    grid_do_in_ellipse(&g, cols / 2.0, rows / 2.0, radius, radius, apply_circular_defect, NULL);

    integrate_params int_params = integrate_params_init();
    int_params.field_func = create_field_D2_over_J(v3d_c(0, 0, 0.5), J, dm, mu);
    int_params.dt = dt;
    int_params.do_cluster = true;
    int_params.output_path = base_path;
    int_params.interval_for_information = 1000;
    int_params.interval_for_raw_grid = 0;
    int_params.interval_for_rgb_grid = 50000;

    int_params.duration = 10 * NS;
    integrate(&g, int_params);

    int_params.duration = 200 * NS;
    int_params.current_func = create_current_stt_ac(2e10, 2e10, 1.0 / 20e-9, 0);
    integrate(&g, int_params);

    grid_free(&g);
    return 0;
}

int main(void) {
    double radii[] = {45, 50, 54, 59, 63, 68, 73, 77, 82, 86, 91};
    int radii_size = sizeof(radii) / sizeof(*radii);
    for (int i = 0; i < radii_size; ++i)
        circular_defect(radii[i]);
    return 0;
}

