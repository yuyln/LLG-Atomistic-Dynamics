#ifndef __GRID_H
#define __GRID_H

#ifndef OPENCL_COMPILATION
#include "v3d.h"
#include <stdint.h>
#endif

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

typedef int8_t pbc_rules;

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

/*
 * Change to a grid struct with global parameters and a array of m?
 */
typedef struct {
    int row, col, depth;
    double exchange, dm, dm_ani, lattice, cubic_ani;
    double mu, alpha, gamma;
    dm_symmetry dm_sym;
    anisotropy ani;
    pinning pin;
    v3d m;
} grid_site;

typedef struct {
    uint64_t rows, cols, depths;
    pbc_rules pbc;
    double time;
    double total_time;
} grid_info;


#endif
