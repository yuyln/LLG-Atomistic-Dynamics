#ifndef __SIMULATION_FUNCS_H
#define __SIMULATION_FUNCS_H

#include "constants.h"
#include "v3d.h"
#include "grid_types.h"
#include "random.h"
#include "tyche_i.c"

typedef struct {
    int rows;
    int cols;
    grid_site_params gs;
    v3d m;
    neighbors_set neigh;
    double time;
    PRIVATE tyche_i_state *state;
#ifdef INCLUDE_DIPOLAR
    pbc_rules pbc;
    GLOBAL v3d *v;
#endif
} parameters;

v3d apply_pbc(GLOBAL v3d *v, pbc_rules pbc, int row, int col, int rows, int cols);
v3d generate_magnetic_field(grid_site_params gs, double time);
current generate_current(grid_site_params gs, double time);
double generate_temperature(grid_site_params gs, double time);

double exchange_energy(parameters param);
double dm_energy(parameters param);
double anisotropy_energy(parameters param);
double cubic_anisotropy_energy(parameters param);
double field_energy(parameters param);
#ifdef INCLUDE_DIPOLAR
double dipolar_energy(parameters param);
#endif
double energy(parameters param);

#ifdef INCLUDE_DIPOLAR
static v3d dipolar_field(parameters param);
#endif
v3d effective_field(parameters param);
v3d dm_dt(parameters param, double dt);
static v3d v3d_dot_grad(v3d v, neighbors_set neigh, double dx, double dy);
v3d step_llg(parameters param, double dt);

double charge_derivative(v3d m, v3d left, v3d right, v3d up, v3d down);
double charge_lattice(v3d m, v3d left, v3d right, v3d up, v3d down);

v3d emergent_magnetic_field_lattice(v3d m, v3d left, v3d right, v3d up, v3d down);
v3d emergent_magnetic_field_derivative(v3d m, v3d left, v3d right, v3d up, v3d down);
v3d emergent_eletric_field(v3d m, v3d left, v3d right, v3d up, v3d down, v3d dmdt, double dx, double dy);
#endif
