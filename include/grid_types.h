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
    double polarization;
} stt_current;

typedef struct {
    v3d p;
    double thickness, theta_sh;
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
    int row, col, depth;
    //double y, x, z;
    double exchange, dm, dm_ani, lattice, cubic_ani;
    double mu, alpha, gamma;
    dm_symmetry dm_sym;
    anisotropy ani;
    pinning pin;
} grid_site_param;

typedef struct {
    unsigned int rows, cols, depths;
    pbc_rules pbc;
} grid_info;

typedef struct {
    v3d left, right, up, down;
#ifndef NBULK
    v3d front, back;
#endif
} neighbors_set;

#endif
