#ifndef __GRID_H
#define __GRID_H

#include <stdbool.h>

#include "v3d.h"

/*typedef enum {
    R_ij         = 0,
    R_ij_CROSS_Z = 1,
} dm_symmetry;*/

typedef enum {
    CUR_NONE = 0,
    CUR_STT = 1,
    CUR_SHE = 2,
    CUR_BOTH = CUR_STT | CUR_SHE //=3
} current_type;

typedef struct {
    v3d m;
    int pbc_x;
    int pbc_y;
} pbc_rules;

typedef struct {
    v3d j;
    double polarization, beta;
} stt_current;

typedef struct {
    v3d p;
    double theta_sh, beta;
} she_current;

typedef struct {
    stt_current stt;
    she_current she;
    current_type type;
} current;

typedef struct {
    v3d dir;
    double ani;
} anisotropy;

typedef struct {
    v3d dir;
    char pinned;
} pinning;

typedef struct {
    v3d dmv_left, dmv_right, dmv_up, dmv_down;
} dm_interaction;

typedef struct {
    int row, col;
    double exchange, lattice, cubic_ani;
    double mu, alpha, gamma;
    anisotropy ani;
    pinning pin;
    dm_interaction dm;
} grid_site_params;

typedef struct {
    unsigned int rows, cols;
    pbc_rules pbc;
} grid_info;

typedef struct {
    v3d left, right, up, down;
} neighbors_set;

typedef struct {
    v3d magnetic_field_finite;
    v3d magnetic_field_lattice;
    v3d electric_field;
    v3d avg_m;

    double charge_center_x;
    double charge_center_y;

    double abs_charge_center_x;
    double abs_charge_center_y;

    double charge_lattice;
    double charge_finite;

    double abs_charge_lattice;
    double abs_charge_finite;

    double exchange_energy;
    double dm_energy;
    double field_energy;
    double anisotropy_energy;
    double cubic_energy;
    double dipolar_energy;
    double energy;

    double D_xx;
    double D_yy;
    double D_xy; //=D_yx
} information_packed;

#endif
