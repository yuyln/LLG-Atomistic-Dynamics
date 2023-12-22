#ifndef __SIMULATION_FUNCS_H
#define __SIMULATION_FUNCS_H

#include "constants.h"
#include "v3d.h"
#include "grid_types.h"

typedef struct {
    grid_site_param gs;
    v3d c, l, r, u, d;
#ifndef NBULK
    v3d f, b;
#endif
    double time;
} parameters;

v3d apply_pbc(GLOBAL v3d *v, matrix_size size, matrix_loc loc, pbc_rules pbc);
v3d get_dm_vec(v3d dr, double dm, dm_symmetry dm_sym);
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

#endif
