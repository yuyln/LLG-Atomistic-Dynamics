#ifndef __GRID_H
#define __GRID_H

#include "v3d.h"

typedef enum {
    R_ij_CROSS_Z = 0,
    R_ij         = 1
} dm_symmetry;

typedef enum {
    CUR_NONE = 0,
    CUR_STT = 1,
    CUR_SHE = 2,
    CUR_BOTH = CUR_STT | CUR_SHE //=3
} current_type;

typedef char pbc_directions;

typedef struct {
    v3d m;
    pbc_directions dirs;
} pbc_rules;

typedef struct {
    v3d j;
    double polarization, beta;
} stt_current;

typedef struct {
    v3d p;
    double thickness, theta_sh, beta;
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
    int row, col;
    //double y, x, z;
    double exchange, dm, dm_ani, lattice, cubic_ani;
    double mu, alpha, gamma;
    dm_symmetry dm_sym;
    anisotropy ani;
    pinning pin;
} grid_site_param;

typedef struct {
    unsigned int rows, cols;
    pbc_rules pbc;
} grid_info;

typedef struct {
    v3d left, right, up, down;
} neighbors_set;

typedef struct {
    v3d avg_B;
    v3d avg_E;
    v3d avg_m;

    double charge_lattice;
    double charge_finite;

    double exchange_energy;
    double dm_energy;
    double field_energy;
    double anisotropy_energy;
    double cubic_energy;
    double energy;
} information_packed;

#endif
