#ifndef __GRID
#define __GRID
#ifndef OPENCLCOMP
#include "vec.h"
#include <stdint.h>
#else
typedef size_t uint64_t;
#endif

typedef enum {
    R_ij, Z_CROSS_R_ij, 
    R_ij_ISOTROPIC_X, Z_CROSS_R_ij_ISOTROPIC_X,
    R_ij_ISOTROPIC_Y, Z_CROSS_R_ij_ISOTROPIC_Y,
    R_ij_CROSS_Z
} DM_TYPE;

typedef enum {
    PBC_XY, PBC_X, PBC_Y, PBC_NONE
} PBC_TYPE;

typedef enum {
    CUR_NONE, CUR_CPP, CUR_STT, CUR_BOTH
} CUR_TYPE;

typedef struct {
    v3d j, p;
    double P, beta, thick, theta_sh;
    CUR_TYPE type;
} current_t;

typedef struct {
    v3d dir;
    PBC_TYPE pbc_type;
} pbc_t;

typedef struct {
    double K_1;
    v3d dir;
} anisotropy_t;

typedef struct {
    char fixed;
    v3d dir;
} pinning_t;

typedef struct {
    int rows, cols;
    uint64_t total;
    double exchange, dm, lattice, cubic_ani;
    double mu_s, lande, avg_spin, alpha, gamma;
    double total_time; //should not be a grid param, however.........
    DM_TYPE dm_type;
    pbc_t pbc;
} grid_param_t;

typedef struct {
    double exchange_mult, dm_mult; //current
    DM_TYPE dm_type;
} region_param_t;

typedef struct {
    grid_param_t param;
    #ifndef OPENCLCOMP
    v3d *grid;
    anisotropy_t *ani;
    pinning_t *pinning;
    region_param_t *regions;
    #else
    v3d grid[TOTAL];
    anisotropy_t ani[TOTAL];
    pinning_t pinning[TOTAL];
    region_param_t regions[TOTAL];
    #endif
} grid_t;

typedef struct {
    double vx;
    double vy;
    double energy;
    double energy_exchange;
    double energy_dm;
    double energy_zeeman;
    double energy_anisotropy;
    double energy_cubic_anisotropy;
    double charge_lattice;
    double charge_finite;
    double charge_cx;
    double charge_cy;
    v3d avg_mag;
} info_pack_t;

#endif
