#ifndef __GRADIENT_DESCENT_H
#define __GRADIENT_DESCENT_H

#include "CL/cl.h"
#include "opencl_wrapper.h"
#include "vec.h"
#include <grid.h>
#include <funcs.h>
#include <helpers.h>
#define PRINT_PARAM 20


void GradientDescentSingle(Grid *g_in, Grid *g_out, double dt, double alpha, double beta, double mass, int steps, Vec field, double T, double T_factor)
{
    int rows = g_in->param.rows;
    int cols = g_in->param.cols;
    double J = fabs(g_in->param.exchange);

    Grid g_aux = InitNullGrid();
    CopyGrid(&g_aux, g_in);
    free(g_aux.grid);

    Vec *g_p = (Vec*)calloc(rows * cols, sizeof(Vec));
    memcpy(g_p, g_in->grid, sizeof(Vec) * rows * cols);

    Vec *g_c = (Vec*)calloc(rows * cols, sizeof(Vec));
    memcpy(g_c, g_in->grid, sizeof(Vec) * rows * cols);

    Vec *g_n = (Vec*)calloc(rows * cols, sizeof(Vec));
    memcpy(g_n, g_in->grid, sizeof(Vec) * rows * cols);

    Vec *g_min = (Vec*)calloc(rows * cols, sizeof(Vec));
    memcpy(g_min, g_in->grid, sizeof(Vec) * rows * cols);

    g_aux.grid = g_min;
    double H_min = Hamiltonian(&g_aux, field);
    for (int i = 0; i < steps; ++i)
    {
	if (i % (steps / PRINT_PARAM) == 0)
	    printf("STEP: %d      H_MIN: %e      T: %e\n", i, H_min / QE, T);

	double H = 0;
        for (int j = 0; j < rows * cols; ++j)
	{
	    g_aux.grid = g_c;
	    Vec vel = GradientDescentVelocity(g_p[j], g_n[j], dt);
	    Vec Heff = GradientDescentForce(j, &g_aux, vel, g_c, field, J, alpha, beta);

	    if (T != 0)
                Heff = VecAdd(Heff, VecScalar(VecFrom(myrandom(), myrandom(), myrandom()), T));

	    g_n[j] = VecAdd(
			    VecSub(VecScalar(g_c[j], 2.0), g_p[j]),
			    VecScalar(Heff, -2.0 * dt * dt / mass)
			   );

	    g_aux.grid = g_n;
	    GridNormalizeI(j, g_aux.grid, g_aux.pinning);
	    H += HamiltonianI(j, g_aux.grid, &g_aux.param, g_aux.ani, g_aux.regions, field);
	}

	memcpy(g_p, g_c, sizeof(Vec) * rows * cols);
	memcpy(g_c, g_n, sizeof(Vec) * rows * cols);
	if (H < H_min)
	{
	    H_min = H;
	    memcpy(g_min, g_n, sizeof(Vec) * rows * cols);
	}
	T *= T_factor;
    }


    FreeGrid(g_out);
    g_aux.grid = g_min;
    CopyGrid(g_out, &g_aux);

    free(g_p);
    free(g_c);
    free(g_n);
    free(g_min);

    g_aux.grid = NULL;
    FreeGrid(&g_aux);
}


void GradientDescentMultiple(Grid *g_in, Grid *g_out, double dt, double alpha, double beta, double mass, int steps, Vec field, double T, double T_factor, int n_threads)
{
    int rows = g_in->param.rows;
    int cols = g_in->param.cols;
    double J = fabs(g_in->param.exchange);

    Grid g_aux = InitNullGrid();
    CopyGrid(&g_aux, g_in);
    free(g_aux.grid);

    Vec *g_p = (Vec*)calloc(rows * cols, sizeof(Vec));
    memcpy(g_p, g_in->grid, sizeof(Vec) * rows * cols);

    Vec *g_c = (Vec*)calloc(rows * cols, sizeof(Vec));
    memcpy(g_c, g_in->grid, sizeof(Vec) * rows * cols);

    Vec *g_n = (Vec*)calloc(rows * cols, sizeof(Vec));
    memcpy(g_n, g_in->grid, sizeof(Vec) * rows * cols);

    Vec *g_min = (Vec*)calloc(rows * cols, sizeof(Vec));
    memcpy(g_min, g_in->grid, sizeof(Vec) * rows * cols);

    g_aux.grid = g_min;
    double H_min = Hamiltonian(&g_aux, field);
    double *Hn = (double*)calloc(n_threads, sizeof(double));

    for (int i = 0; i < steps; ++i)
    {
	if (i % (steps / PRINT_PARAM) == 0)
	    printf("STEP: %d      H_MIN: %e      T: %e\n", i, H_min / QE, T);

	memset(Hn, 0, n_threads * sizeof(double));
	double H = 0.0;
	int j;
	#pragma omp parallel for num_threads(n_threads)
        for (j = 0; j < rows * cols; ++j)
	{
	    g_aux.grid = g_c;
	    Vec vel = GradientDescentVelocity(g_p[j], g_n[j], dt);
	    Vec Heff = GradientDescentForce(j, &g_aux, vel, g_c, field, J, alpha, beta);

	    if (T != 0)
	        Heff = VecAdd(Heff, VecScalar(VecFrom(myrandom(), myrandom(), myrandom()), T));

	    g_n[j] = VecAdd(
			    VecSub(VecScalar(g_c[j], 2.0), g_p[j]),
			    VecScalar(Heff, -2.0 * dt * dt / mass)
			   );

	    g_aux.grid = g_n;
	    GridNormalizeI(j, g_aux.grid, g_aux.pinning);
	    Hn[omp_get_thread_num()] += HamiltonianI(j, g_aux.grid, &g_aux.param, g_aux.ani, g_aux.regions, field);
	}

	for (int j = 0; j < n_threads; ++j)
	    H += Hn[j];

	memcpy(g_p, g_c, sizeof(Vec) * rows * cols);
	memcpy(g_c, g_n, sizeof(Vec) * rows * cols);
	if (H < H_min)
	{
	    H_min = H;
	    memcpy(g_min, g_n, sizeof(Vec) * rows * cols);
	}
	T *= T_factor;
    }


    FreeGrid(g_out);
    g_aux.grid = g_min;
    CopyGrid(g_out, &g_aux);

    free(g_p);
    free(g_c);
    free(g_n);
    free(g_min);
    free(Hn);

    g_aux.grid = NULL;
    FreeGrid(&g_aux);
}

void GradientDescentGPU(Grid *g_in, Grid *g_out, double dt, double alpha, double beta, double mass, int steps, Vec field, double T, double T_factor, GPU *gpu)
{
    int rows = g_in->param.rows;
    int cols = g_in->param.cols;
    double J = fabs(g_in->param.exchange);

    Grid g_aux = InitNullGrid();
    CopyGrid(&g_aux, g_in);

    cl_mem g_aux_buffer = CreateBuffer(FindGridSize(&g_aux), gpu->ctx, CL_MEM_READ_WRITE);
    cl_mem g_p_buffer = CreateBuffer(sizeof(Vec) * rows * cols, gpu->ctx, CL_MEM_READ_WRITE);
    cl_mem g_c_buffer = CreateBuffer(sizeof(Vec) * rows * cols, gpu->ctx, CL_MEM_READ_WRITE);
    cl_mem g_n_buffer = CreateBuffer(sizeof(Vec) * rows * cols, gpu->ctx, CL_MEM_READ_WRITE);
    cl_mem g_min_buffer = CreateBuffer(sizeof(Vec) * rows * cols, gpu->ctx, CL_MEM_READ_WRITE);

    WriteBuffer(g_p_buffer, g_in->grid, sizeof(Vec) * rows * cols, 0, gpu->queue);
    WriteBuffer(g_c_buffer, g_in->grid, sizeof(Vec) * rows * cols, 0, gpu->queue);
    WriteBuffer(g_n_buffer, g_in->grid, sizeof(Vec) * rows * cols, 0, gpu->queue);
    WriteBuffer(g_min_buffer, g_in->grid, sizeof(Vec) * rows * cols, 0, gpu->queue);

    cl_mem H_buffer = CreateBuffer(sizeof(double) * rows * cols, gpu->ctx, CL_MEM_READ_WRITE);
    double *H = (double*)calloc(rows * cols, sizeof(double));


    double H_min = Hamiltonian(&g_aux, field);
    free(g_aux.grid);

    WriteFullGridBuffer(gpu->queue, g_aux_buffer, &g_aux);

    SetKernelArg(gpu->kernels[4], 0, sizeof(cl_mem), &g_aux_buffer);
    SetKernelArg(gpu->kernels[4], 1, sizeof(cl_mem), &g_p_buffer);
    SetKernelArg(gpu->kernels[4], 2, sizeof(cl_mem), &g_c_buffer);
    SetKernelArg(gpu->kernels[4], 3, sizeof(cl_mem), &g_n_buffer);
    SetKernelArg(gpu->kernels[4], 4, sizeof(double), &dt);
    SetKernelArg(gpu->kernels[4], 5, sizeof(double), &alpha);
    SetKernelArg(gpu->kernels[4], 6, sizeof(double), &beta);
    SetKernelArg(gpu->kernels[4], 7, sizeof(double), &mass);
    SetKernelArg(gpu->kernels[4], 9, sizeof(cl_mem), &H_buffer);
    SetKernelArg(gpu->kernels[4], 11, sizeof(double), &J);
    SetKernelArg(gpu->kernels[4], 12, sizeof(Vec), &field);

    srand(time(NULL));
    size_t global = rows * cols;
    size_t local = gcd(global, 32);
    for (int i = 0; i < steps; ++i)
    {
	SetKernelArg(gpu->kernels[4], 8, sizeof(double), &T);
	if (i % (steps / PRINT_PARAM) == 0)
	    printf("STEP: %d      H_MIN: %e      T: %e\n", i, H_min / QE, T);

	int seed = rand();
	SetKernelArg(gpu->kernels[4], 10, sizeof(int), &seed);
	EnqueueND(gpu->queue, gpu->kernels[4], 1, NULL, &global, &local);
	ReadBuffer(H_buffer, H, sizeof(double) * rows * cols, 0, gpu->queue);

	double Hl = 0.0;
	for (int j = 0; j < rows * cols; ++j)
	    Hl += H[j];

	//memcpy(g_p, g_c, sizeof(Vec) * rows * cols);
	//memcpy(g_c, g_n, sizeof(Vec) * rows * cols);
	SetKernelArg(gpu->kernels[5], 0, sizeof(cl_mem), &g_p_buffer);
	SetKernelArg(gpu->kernels[5], 1, sizeof(cl_mem), &g_c_buffer);
	EnqueueND(gpu->queue, gpu->kernels[5], 1, NULL, &global, &local);

	SetKernelArg(gpu->kernels[5], 0, sizeof(cl_mem), &g_c_buffer);
	SetKernelArg(gpu->kernels[5], 1, sizeof(cl_mem), &g_n_buffer);
	EnqueueND(gpu->queue, gpu->kernels[5], 1, NULL, &global, &local);

	if (Hl < H_min)
	{
	    H_min = Hl;
	    //memcpy(g_min, g_n, sizeof(Vec) * rows * cols);
	    SetKernelArg(gpu->kernels[5], 0, sizeof(cl_mem), &g_min_buffer);
	    SetKernelArg(gpu->kernels[5], 1, sizeof(cl_mem), &g_n_buffer);
	    EnqueueND(gpu->queue, gpu->kernels[5], 1, NULL, &global, &local);
	}
	T *= T_factor;
    }


    Vec *g_min = (Vec*)calloc(rows * cols, sizeof(Vec));
    ReadBuffer(g_min_buffer, g_min, sizeof(Vec) * rows * cols, 0, gpu->queue);
    g_aux.grid = g_min;
    FreeGrid(g_out);
    CopyGrid(g_out, &g_aux);
    g_aux.grid = NULL;
    FreeGrid(&g_aux);

    free(H);
    free(g_min);
    PrintCLError(stderr, clReleaseMemObject(H_buffer), "Error releasing H_buffer");
    PrintCLError(stderr, clReleaseMemObject(g_p_buffer), "Error releasing g_p_buffer");
    PrintCLError(stderr, clReleaseMemObject(g_c_buffer), "Error releasing g_c_buffer");
    PrintCLError(stderr, clReleaseMemObject(g_n_buffer), "Error releasing g_n_buffer");
    PrintCLError(stderr, clReleaseMemObject(g_aux_buffer), "Error releasing g_aux_buffer"); 
    PrintCLError(stderr, clReleaseMemObject(g_min_buffer), "Error releasing g_min_buffer");
}

void GradientDescent(Grid *g_in, Grid *g_out, double dt, double alpha, double beta, double mass, int steps, Vec field, double T, double T_factor, GPU *gpu, bool use_gpu, int n_threads)
{
    if (use_gpu)
	GradientDescentGPU(g_in, g_out, dt, alpha, beta, mass, steps, field, T, T_factor, gpu);
    else if (n_threads > 1)
	GradientDescentMultiple(g_in, g_out, dt, alpha, beta, mass, steps, field, T, T_factor, n_threads);
    else
	GradientDescentSingle(g_in, g_out, dt, alpha, beta, mass, steps, field, T, T_factor);
}
#endif
