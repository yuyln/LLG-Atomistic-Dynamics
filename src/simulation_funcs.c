#include "simulation_funcs.h"

v3d apply_pbc(GLOBAL v3d *v, matrix_size size, matrix_loc loc, pbc_rules pbc) {
    int pbc_x = pbc.dirs & (1 << 0);
    int pbc_y = pbc.dirs & (1 << 1);
    int pbc_z = pbc.dirs & (1 << 2);

    if (loc.dim[0] >= size.dim[0] || loc.dim[0] < 0) {
        if (!pbc_y)
            return pbc.m;
        loc.dim[0] = ((loc.dim[0] % size.dim[0]) + size.dim[0]) % size.dim[0];
    }

    if (loc.dim[1] >= size.dim[1] || loc.dim[1] < 0) {
        if (!pbc_x)
            return pbc.m;
        loc.dim[1] = ((loc.dim[1] % size.dim[1]) + size.dim[1]) % size.dim[1];
    }

    if (loc.dim[2] >= size.dim[2] || loc.dim[2] < 0) {
        if (!pbc_z)
            return pbc.m;
        loc.dim[2] = ((loc.dim[2] % size.dim[2]) + size.dim[2]) % size.dim[2];
    }

    return v[LOC(loc.row, loc.col, loc.depth, size.rows, size.cols)];
}

/*
v3d generate_magnetic_field(grid_site_param gs, double time);
current generate_current(grid_site_param gs, double time);

double exchange_energy(parameters param);
double dm_energy(parameters param);
double anisotropy_energy(parameters param);
double cubic_anisotropy_energy(parameters param);
double field_energy(parameters param);
double energy(parameters param);

v3d effective_field(parameters param);
v3d dm_dt(parameters param);
v3d step(parameters param, double dt);

double charge_derivative(parameters param);
double charge_lattice(parameters param);
*/
