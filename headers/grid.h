#ifndef __GRID
#define __GRID
#include <vec.h>

typedef enum
{
    R_ij, Z_CROSS_R_ij
} DM_TYPE;

typedef enum
{
    PBC_XY, PBC_X, PBC_Y, PBC_NONE
} PBC_TYPE;

typedef enum
{
    CUR_NONE, CUR_CPP, CUR_STT, CUR_BOTH
} CUR_TYPE;

typedef struct
{
    Vec j;
    double p, beta, thick;
    CUR_TYPE type;
} Current;

typedef struct
{
    Vec dir;
    PBC_TYPE pbc_type;
} PBC;

typedef struct
{
    double K_1;
    Vec dir;
} Anisotropy;

typedef struct
{
    char fixed;
    Vec dir;
} Pinning;

typedef struct
{
    int rows, cols;
    size_t total;
    double exchange, dm, lattice, cubic_ani;
    double mu_s, lande, avg_spin, alpha, gamma;
    double total_time; //should not be a grid param, however.........
    DM_TYPE dm_type;
    PBC pbc;
} GridParam;

typedef struct
{
    double exchange_mult, dm_mult, field_mult; //current
    DM_TYPE dm_type;
} RegionParam;

typedef struct
{
    GridParam param;
    #ifndef OPENCLCOMP
    Vec *grid;
    Anisotropy *ani;
    Pinning *pinning;
    RegionParam *regions;
    #else
    Vec grid[TOTAL];
    Anisotropy ani[TOTAL];
    Pinning pinning[TOTAL];
    RegionParam regions[TOTAL];
    #endif
} Grid;

#endif