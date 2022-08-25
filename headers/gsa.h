#ifndef __GSA
#define __GSA

#include <helpers.h>

typedef struct
{
    double qA, qT, qV, T0;
    size_t inner_loop, outer_loop, print_param;
} GSAParam;

void GSA(GSAParam param, Grid* g_in, Grid* g_out, Vec field)
{
    Grid g_min = InitNullGrid(),
         g_old = InitNullGrid();
    
    {
        printf("qA: %e\n", param.qA);
        printf("qT: %e\n", param.qT);
        printf("qV: %e\n", param.qV);
        printf("T0: %e\n", param.T0);
        printf("outer_loop: %zu\n", param.outer_loop);
        printf("inner_loop: %zu\n", param.inner_loop);
    }

    CopyGrid(&g_min, g_in);
    CopyGrid(&g_old, g_in);
    CopyGrid(g_out, g_in);

    double H_old = Hamiltonian(&g_old, field) / g_in->param.exchange,
           H_new = Hamiltonian(g_out, field) / g_in->param.exchange,
           H_min = Hamiltonian(&g_min, field) / g_in->param.exchange;

    double qA1 = param.qA - 1.0,
           qV1 = param.qV - 1.0,
           qT1 = param.qT - 1.0,
           oneqA1 = 1.0 / qA1,
           exp1 = 2.0 / (3.0 - param.qV),
           exp2 = 1.0 / qV1 - 0.5;
    
    for (size_t outer = 1; outer <= param.outer_loop; ++outer)
    {
        srand(time(NULL));
        double t = 0.0,
               Tqt = param.T0 * (pow(2.0, qT1) - 1.0);
        for (size_t inner = 1; inner <= param.inner_loop; ++inner)
        {
            t += 1.0;
            double T = Tqt / (pow(t + 1.0, qT1) - 1.0);
            if (inner % (param.inner_loop / param.print_param) == 0)
                printf("outer: %zu inner: %zu H_min: %e T: %e\n", outer, inner, H_min * g_in->param.exchange, T);
            
            for (size_t I = 0; I < g_in->param.total; ++I)
            {
                double R = myrandom();
                double delta = 1.0 / pow(1.0 + qV1 * R * R / pow(T, exp1), exp2);
                if (myrandom() < 0.5)
                    delta = -delta;
                g_out->grid[I].x = g_old.grid[I].x + delta;

                //------------------------------

                R = myrandom();
                delta = 1.0 / pow(1.0 + qV1 * R * R / pow(T, exp1), exp2);
                if (myrandom() < 0.5)
                    delta = -delta;
                g_out->grid[I].y = g_old.grid[I].y + delta;

                //------------------------------

                R = myrandom();
                delta = 1.0 / pow(1.0 + qV1 * R * R / pow(T, exp1), exp2);
                if (myrandom() < 0.5)
                    delta = -delta;
                g_out->grid[I].z = g_old.grid[I].z + delta;

                GridNormalizeI(I, g_out);
            }
            
            H_new = Hamiltonian(g_out, field) / g_in->param.exchange;

            if (H_new <= H_min)
            {
                H_min = H_new;
                CopySpinsToExistingGrid(&g_min, g_out);   
            }

            if (H_new <= H_old)
            {
                H_old = H_new;
                CopySpinsToExistingGrid(&g_old, g_out);
            }
            else
            {
                double df = H_new - H_old;
                double pqa = 1.0 / pow(1.0 + qA1 * df / T, oneqA1);
                if (myrandom() < pqa)
                {
                    H_old = H_new;
                    CopySpinsToExistingGrid(&g_old, g_out);
                }
            }
        }
    }
    CopyGrid(g_out, &g_min);
    FreeGrid(&g_min);
    FreeGrid(&g_old);
}


#endif