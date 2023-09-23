#ifndef __GRID
#define __GRID
#ifndef OPENCLCOMP
#include "./headers/vec.h"
#endif

typedef enum {
    R_ij, Z_CROSS_R_ij
} DM_TYPE;

typedef enum {
    pbc_t_XY, pbc_t_X, pbc_t_Y, pbc_t_NONE
} PBC_TYPE;

typedef enum {
    CUR_NONE, CUR_CPP, CUR_STT, CUR_BOTH
} CUR_TYPE;

typedef struct {
    v3d j;
    double p, beta, thick;
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
    size_t total;
    double exchange, dm, lattice, cubic_ani;
    double mu_s, lande, avg_spin, alpha, gamma;
    double total_time; //should not be a grid param, however.........
    DM_TYPE dm_type;
    pbc_t pbc;
} grid_param_t;

typedef struct {
    double exchange_mult, dm_mult, field_mult; //current
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

#endif
