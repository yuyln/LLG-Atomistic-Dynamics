#ifndef __GRID
#define __GRID
#include <vec.h>

typedef enum
{
    R_ij, Z_CROSS_R_ij
} DM_TYPE;

typedef enum
{
    XY, X, Y, NONE
} PBC_TYPE;

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
    double mu_s, lande, avg_spin;
    DM_TYPE dm_type;
    PBC pbc;
} GridParam;

typedef struct
{
    GridParam param;
    Vec *grid;
    Anisotropy *ani;
    Pinning *pinning;
} Grid;

#endif