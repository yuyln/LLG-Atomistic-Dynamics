#ifndef __GSA
#define __GSA

#include <helpers.h>

typedef struct
{
    double qA, qT, qV, T0;
    size_t inner_loop, outer_loop, print_param;
} GSAParam;

void GSA(GSAParam param, Grid* g_in, Grid* g_out)
{
    (void)param;
    Grid g_min = InitNullGrid(),
         g_old = InitNullGrid();
    
    CopyGrid(&g_min, g_in);
    CopyGrid(&g_old, g_in);


    CopyGrid(g_out, &g_min);
    FreeGrid(&g_min);
    FreeGrid(&g_old);
}


#endif