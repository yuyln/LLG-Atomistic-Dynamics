#ifndef __GSA
#define __GSA

#include <helpers.h>

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

    double norm_factor = g_in->param.exchange * (g_in->param.exchange > 0? 1.0: -1.0);

    double H_old = Hamiltonian(&g_old, field),
           H_new = Hamiltonian(g_out, field),
           H_min = Hamiltonian(&g_min, field);

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
                printf("outer: %zu inner: %zu H_min: %e T: %e\n", outer, inner, H_min, T);
            
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

                GridNormalizeI(I, g_out->grid, g_out->pinning);
            }
            
            H_new = Hamiltonian(g_out, field);

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
                double df_norm = (H_new - H_old) / norm_factor;
                double pqa = 1.0 / pow(1.0 + qA1 * df_norm / T, oneqA1);
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

void GSAGPU(GSAParam param, Grid* g_in, Grid* g_out, Vec field, GPU *gpu)
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

    double norm_factor = g_in->param.exchange * (g_in->param.exchange > 0? 1.0: -1.0);

    double H_old = Hamiltonian(&g_old, field),
           H_new = Hamiltonian(g_out, field),
           H_min = Hamiltonian(&g_min, field);

    FreeGrid(&g_min);
    FreeGrid(&g_old);

    double qA1 = param.qA - 1.0,
           qV1 = param.qV - 1.0,
           qT1 = param.qT - 1.0,
           oneqA1 = 1.0 / qA1,
           exp1 = 2.0 / (3.0 - param.qV),
           exp2 = 1.0 / qV1 - 0.5;
    
    cl_mem g_min_buffer = CreateBuffer(FindGridSize(g_in), gpu->ctx, CL_MEM_READ_WRITE), 
           g_old_buffer = CreateBuffer(FindGridSize(g_in), gpu->ctx, CL_MEM_READ_WRITE),
           g_out_buffer = CreateBuffer(FindGridSize(g_in), gpu->ctx, CL_MEM_READ_WRITE),
           ham_buffer = CreateBuffer(sizeof(double) * g_in->param.total, gpu->ctx, CL_MEM_READ_WRITE);
    
    WriteFullGridBuffer(gpu->queue, g_min_buffer, g_in);
    WriteFullGridBuffer(gpu->queue, g_old_buffer, g_in);
    WriteFullGridBuffer(gpu->queue, g_out_buffer, g_in);

    double* ham_gpu = (double*)calloc(g_in->param.total, sizeof(double));


    size_t global = g_in->param.total,
           local = gcd(global, 32);

    SetKernelArg(gpu->kernels[0], 0, sizeof(cl_mem), &g_out_buffer);
    SetKernelArg(gpu->kernels[0], 1, sizeof(cl_mem), &g_old_buffer);
    SetKernelArg(gpu->kernels[0], 3, sizeof(double), &qV1);
    SetKernelArg(gpu->kernels[0], 4, sizeof(double), &exp1);
    SetKernelArg(gpu->kernels[0], 5, sizeof(double), &exp2);
    
    SetKernelArg(gpu->kernels[1], 0, sizeof(cl_mem), &g_out_buffer);
    SetKernelArg(gpu->kernels[1], 1, sizeof(cl_mem), &ham_buffer);
    SetKernelArg(gpu->kernels[1], 2, sizeof(Vec) ,&field);

    SetKernelArg(gpu->kernels[2], 1, sizeof(cl_mem), &g_out_buffer);

    for (size_t outer = 1; outer <= param.outer_loop; ++outer)
    {
        srand(time(NULL));
        double t = 0.0,
               Tqt = param.T0 * (pow(2.0, qT1) - 1.0);
        for (size_t inner = 1; inner <= param.inner_loop; ++inner)
        {
            t += 1.0;
            double T = Tqt / (pow(t + 1.0, qT1) - 1.0);
            SetKernelArg(gpu->kernels[0], 2, sizeof(double) ,&T);
            if (inner % (param.inner_loop / param.print_param) == 0)
                printf("outer: %zu inner: %zu H_min: %e T: %e\n", outer, inner, H_min, T);
            int seed = rand();
            SetKernelArg(gpu->kernels[0], 6, sizeof(int) ,&seed);
            EnqueueND(gpu->queue, gpu->kernels[0], 1, NULL, &global, &local);
            //Finish(gpu->queue);

            EnqueueND(gpu->queue, gpu->kernels[1], 1, NULL, &global, &local);
            //Finish(gpu->queue);

            ReadBuffer(ham_buffer, ham_gpu, sizeof(double) * g_in->param.total, 0, gpu->queue);

            double HH = 0.0;
            for (size_t i = 0; i < g_in->param.total; ++i)
                HH += ham_gpu[i];
            
            H_new = HH;
            
            if (H_new <= H_min)
            {
                H_min = H_new;
                SetKernelArg(gpu->kernels[2], 0, sizeof(cl_mem), &g_min_buffer);
                EnqueueND(gpu->queue, gpu->kernels[2], 1, NULL, &global, &local);
                //Finish(gpu->queue);
            }

            if (H_new <= H_old)
            {
                H_old = H_new;
                SetKernelArg(gpu->kernels[2], 0, sizeof(cl_mem), &g_old_buffer);
                EnqueueND(gpu->queue, gpu->kernels[2], 1, NULL, &global, &local);
                //Finish(gpu->queue);
            }
            else
            {
                double df_norm = (H_new - H_old) / norm_factor;
                double pqa = 1.0 / pow(1.0 + qA1 * df_norm / T, oneqA1);
                if (myrandom() < pqa)
                {
                    H_old = H_new;
                    SetKernelArg(gpu->kernels[2], 0, sizeof(cl_mem), &g_old_buffer);
                    EnqueueND(gpu->queue, gpu->kernels[2], 1, NULL, &global, &local);
                    //Finish(gpu->queue);
                }
            }
        }
    }
    ReadFullGridBuffer(gpu->queue, g_min_buffer, g_out);

    free(ham_gpu);
    PrintCLError(stderr, clReleaseMemObject(g_min_buffer), "Could not release g_min_buffer");
    PrintCLError(stderr, clReleaseMemObject(g_out_buffer), "Could not release g_out_buffer");
    PrintCLError(stderr, clReleaseMemObject(g_old_buffer), "Could not release g_old_buffer");
    PrintCLError(stderr, clReleaseMemObject(ham_buffer), "Could not release ham_buffer");
}
#endif
