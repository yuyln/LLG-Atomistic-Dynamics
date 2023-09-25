#ifndef __GRADIENT_DESCENT_H
#define __GRADIENT_DESCENT_H

#include "opencl_wrapper.h"
#include "grid.h"
#include "funcs.h"
#include "helpers.h"
#define PRINT_PARAM 20


void gradient_descent_single(grid_t *g_in, grid_t *g_out, double dt, double alpha, double beta, double mass, int steps, v3d field, double T, double T_factor) {
    int rows = g_in->param.rows;
    int cols = g_in->param.cols;
    double J = fabs(g_in->param.exchange);

    grid_t g_aux = grid_init_null();
    grid_copy(&g_aux, g_in);
    free(g_aux.grid);

    v3d *g_p = (v3d*)calloc(rows * cols, sizeof(v3d));
    memcpy(g_p, g_in->grid, sizeof(v3d) * rows * cols);

    v3d *g_c = (v3d*)calloc(rows * cols, sizeof(v3d));
    memcpy(g_c, g_in->grid, sizeof(v3d) * rows * cols);

    v3d *g_n = (v3d*)calloc(rows * cols, sizeof(v3d));
    memcpy(g_n, g_in->grid, sizeof(v3d) * rows * cols);

    v3d *g_min = (v3d*)calloc(rows * cols, sizeof(v3d));
    memcpy(g_min, g_in->grid, sizeof(v3d) * rows * cols);

    g_aux.grid = g_min;
    double H_min = hamiltonian(&g_aux, field);
    for (int i = 0; i < steps; ++i) { 
        double local_T = 0;

	if (i >= steps / 2) {
	    local_T = T;
	    T *= T_factor;
        }

	if (i % (steps / PRINT_PARAM) == 0)
	    printf("STEP: %d      H_MIN: %e      T: %e\n", i, H_min / QE, local_T);

	double H = 0;

        for (int j = 0; j < rows * cols; ++j) {
	    g_aux.grid = g_c;
	    v3d vel = gradient_descente_velocity(g_p[j], g_n[j], dt);
	    v3d Heff = gradient_descent_force(j, &g_aux, vel, g_c, field, J, alpha, beta);

	    if (T != 0)
                Heff = v3d_add(Heff, v3d_scalar(v3d_c(2.0 * rand_double() - 1.0, 2.0 * rand_double() - 1.0, 2.0 * rand_double() - 1.0), local_T));

	    g_n[j] = v3d_add(
			    v3d_sub(v3d_scalar(g_c[j], 2.0), g_p[j]),
			    v3d_scalar(Heff, -dt * dt / mass)
			   );

	    g_aux.grid = g_n;
	    grid_normalize(j, g_aux.grid, g_aux.pinning);
	    H += hamiltonian_I(j, g_aux.grid, &g_aux.param, g_aux.ani, g_aux.regions, field);
	}

	memcpy(g_p, g_c, sizeof(v3d) * rows * cols);
	memcpy(g_c, g_n, sizeof(v3d) * rows * cols);
	if (H < H_min) {
	    H_min = H;
	    memcpy(g_min, g_n, sizeof(v3d) * rows * cols);
	}
    }


    grid_free(g_out);
    g_aux.grid = g_min;
    grid_copy(g_out, &g_aux);

    free(g_p);
    free(g_c);
    free(g_n);
    free(g_min);

    g_aux.grid = NULL;
    grid_free(&g_aux);
}


void gradient_descent_multiple(grid_t *g_in, grid_t *g_out, double dt, double alpha, double beta, double mass, int steps, v3d field, double T, double T_factor, int n_threads) {
    int rows = g_in->param.rows;
    int cols = g_in->param.cols;
    double J = fabs(g_in->param.exchange);

    grid_t g_aux = grid_init_null();
    grid_copy(&g_aux, g_in);
    free(g_aux.grid);

    v3d *g_p = (v3d*)calloc(rows * cols, sizeof(v3d));
    memcpy(g_p, g_in->grid, sizeof(v3d) * rows * cols);

    v3d *g_c = (v3d*)calloc(rows * cols, sizeof(v3d));
    memcpy(g_c, g_in->grid, sizeof(v3d) * rows * cols);

    v3d *g_n = (v3d*)calloc(rows * cols, sizeof(v3d));
    memcpy(g_n, g_in->grid, sizeof(v3d) * rows * cols);

    v3d *g_min = (v3d*)calloc(rows * cols, sizeof(v3d));
    memcpy(g_min, g_in->grid, sizeof(v3d) * rows * cols);

    g_aux.grid = g_min;
    double H_min = hamiltonian(&g_aux, field);
    double *Hn = (double*)calloc(n_threads, sizeof(double));

    for (int i = 0; i < steps; ++i) {
        double local_T = 0;

	if (i >= steps / 2) {
	    local_T = T;
	    T *= T_factor;
        }

	if (i % (steps / PRINT_PARAM) == 0)
	    printf("STEP: %d      H_MIN: %e      T: %e\n", i, H_min / QE, local_T);

	memset(Hn, 0, n_threads * sizeof(double));

	if (i >= steps / 2) {
	    local_T = T;
	    T *= T_factor;
	}

	double H = 0.0;
	int j;
	#pragma omp parallel for num_threads(n_threads)
        for (j = 0; j < rows * cols; ++j) {
	    g_aux.grid = g_c;
	    v3d vel = gradient_descente_velocity(g_p[j], g_n[j], dt);
	    v3d Heff = gradient_descent_force(j, &g_aux, vel, g_c, field, J, alpha, beta);

	    if (T != 0)
                Heff = v3d_add(Heff, v3d_scalar(v3d_c(2.0 * rand_double() - 1.0, 2.0 * rand_double() - 1.0, 2.0 * rand_double() - 1.0), local_T));

	    g_n[j] = v3d_add(
			    v3d_sub(v3d_scalar(g_c[j], 2.0), g_p[j]),
			    v3d_scalar(Heff, -dt * dt / mass)
			   );

	    g_aux.grid = g_n;
	    grid_normalize(j, g_aux.grid, g_aux.pinning);
	    Hn[omp_get_thread_num()] += hamiltonian_I(j, g_aux.grid, &g_aux.param, g_aux.ani, g_aux.regions, field);
	}

	for (int j = 0; j < n_threads; ++j)
	    H += Hn[j];

	memcpy(g_p, g_c, sizeof(v3d) * rows * cols);
	memcpy(g_c, g_n, sizeof(v3d) * rows * cols);
	if (H < H_min) {
	    H_min = H;
	    memcpy(g_min, g_n, sizeof(v3d) * rows * cols);
	}
    }


    grid_free(g_out);
    g_aux.grid = g_min;
    grid_copy(g_out, &g_aux);

    free(g_p);
    free(g_c);
    free(g_n);
    free(g_min);
    free(Hn);

    g_aux.grid = NULL;
    grid_free(&g_aux);
}

void gradient_descent_gpu(grid_t *g_in, grid_t *g_out, double dt, double alpha, double beta, double mass, int steps, v3d field, double T, double T_factor, gpu_t *gpu) {
    int rows = g_in->param.rows;
    int cols = g_in->param.cols;
    double J = fabs(g_in->param.exchange);

    grid_t g_aux = grid_init_null();
    grid_copy(&g_aux, g_in);

    cl_mem g_aux_buffer = clw_create_buffer(find_grid_size_bytes(&g_aux), gpu->ctx, CL_MEM_READ_WRITE);
    cl_mem g_p_buffer = clw_create_buffer(sizeof(v3d) * rows * cols, gpu->ctx, CL_MEM_READ_WRITE);
    cl_mem g_c_buffer = clw_create_buffer(sizeof(v3d) * rows * cols, gpu->ctx, CL_MEM_READ_WRITE);
    cl_mem g_n_buffer = clw_create_buffer(sizeof(v3d) * rows * cols, gpu->ctx, CL_MEM_READ_WRITE);
    cl_mem g_min_buffer = clw_create_buffer(sizeof(v3d) * rows * cols, gpu->ctx, CL_MEM_READ_WRITE);

    clw_write_buffer(g_p_buffer, g_in->grid, sizeof(v3d) * rows * cols, 0, gpu->queue);
    clw_write_buffer(g_c_buffer, g_in->grid, sizeof(v3d) * rows * cols, 0, gpu->queue);
    clw_write_buffer(g_n_buffer, g_in->grid, sizeof(v3d) * rows * cols, 0, gpu->queue);
    clw_write_buffer(g_min_buffer, g_in->grid, sizeof(v3d) * rows * cols, 0, gpu->queue);

    cl_mem H_buffer = clw_create_buffer(sizeof(double) * rows * cols, gpu->ctx, CL_MEM_READ_WRITE);
    double *H = (double*)calloc(rows * cols, sizeof(double));


    double H_min = hamiltonian(&g_aux, field);
    free(g_aux.grid);

    full_grid_write_buffer(gpu->queue, g_aux_buffer, &g_aux);

    clw_set_kernel_arg(gpu->kernels[4], 0, sizeof(cl_mem), &g_aux_buffer);
    clw_set_kernel_arg(gpu->kernels[4], 1, sizeof(cl_mem), &g_p_buffer);
    clw_set_kernel_arg(gpu->kernels[4], 2, sizeof(cl_mem), &g_c_buffer);
    clw_set_kernel_arg(gpu->kernels[4], 3, sizeof(cl_mem), &g_n_buffer);
    clw_set_kernel_arg(gpu->kernels[4], 4, sizeof(double), &dt);
    clw_set_kernel_arg(gpu->kernels[4], 5, sizeof(double), &alpha);
    clw_set_kernel_arg(gpu->kernels[4], 6, sizeof(double), &beta);
    clw_set_kernel_arg(gpu->kernels[4], 7, sizeof(double), &mass);
    clw_set_kernel_arg(gpu->kernels[4], 9, sizeof(cl_mem), &H_buffer);
    clw_set_kernel_arg(gpu->kernels[4], 11, sizeof(double), &J);
    clw_set_kernel_arg(gpu->kernels[4], 12, sizeof(v3d), &field);

    srand(time(NULL));
    uint64_t global = rows * cols;
    uint64_t local = gcd(global, 32);
    for (int i = 0; i < steps; ++i) {
        double local_T = 0;

	if (i >= steps / 2) {
	    local_T = T;
	    T *= T_factor;
        }
	clw_set_kernel_arg(gpu->kernels[4], 8, sizeof(double), &local_T);

	if (i % (steps / PRINT_PARAM) == 0)
	    printf("STEP: %d      H_MIN: %e      T: %e\n", i, H_min / QE, local_T);

	int seed = rand();
	clw_set_kernel_arg(gpu->kernels[4], 10, sizeof(int), &seed);
	clw_enqueue_nd(gpu->queue, gpu->kernels[4], 1, NULL, &global, &local);
	clw_read_buffer(H_buffer, H, sizeof(double) * rows * cols, 0, gpu->queue);

	double Hl = 0.0;
	for (int j = 0; j < rows * cols; ++j)
	    Hl += H[j];

	//memcpy(g_p, g_c, sizeof(v3d) * rows * cols);
	//memcpy(g_c, g_n, sizeof(v3d) * rows * cols);
	clw_set_kernel_arg(gpu->kernels[5], 0, sizeof(cl_mem), &g_p_buffer);
	clw_set_kernel_arg(gpu->kernels[5], 1, sizeof(cl_mem), &g_c_buffer);
	clw_enqueue_nd(gpu->queue, gpu->kernels[5], 1, NULL, &global, &local);

	clw_set_kernel_arg(gpu->kernels[5], 0, sizeof(cl_mem), &g_c_buffer);
	clw_set_kernel_arg(gpu->kernels[5], 1, sizeof(cl_mem), &g_n_buffer);
	clw_enqueue_nd(gpu->queue, gpu->kernels[5], 1, NULL, &global, &local);

	if (Hl < H_min) {
	    H_min = Hl;
	    //memcpy(g_min, g_n, sizeof(v3d) * rows * cols);
	    clw_set_kernel_arg(gpu->kernels[5], 0, sizeof(cl_mem), &g_min_buffer);
	    clw_set_kernel_arg(gpu->kernels[5], 1, sizeof(cl_mem), &g_n_buffer);
	    clw_enqueue_nd(gpu->queue, gpu->kernels[5], 1, NULL, &global, &local);
	}
    }


    v3d *g_min = (v3d*)calloc(rows * cols, sizeof(v3d));
    clw_read_buffer(g_min_buffer, g_min, sizeof(v3d) * rows * cols, 0, gpu->queue);
    g_aux.grid = g_min;
    grid_free(g_out);
    grid_copy(g_out, &g_aux);
    g_aux.grid = NULL;
    grid_free(&g_aux);

    free(H);
    free(g_min);
    clw_print_cl_error(stderr, clReleaseMemObject(H_buffer), "Error releasing H_buffer");
    clw_print_cl_error(stderr, clReleaseMemObject(g_p_buffer), "Error releasing g_p_buffer");
    clw_print_cl_error(stderr, clReleaseMemObject(g_c_buffer), "Error releasing g_c_buffer");
    clw_print_cl_error(stderr, clReleaseMemObject(g_n_buffer), "Error releasing g_n_buffer");
    clw_print_cl_error(stderr, clReleaseMemObject(g_aux_buffer), "Error releasing g_aux_buffer"); 
    clw_print_cl_error(stderr, clReleaseMemObject(g_min_buffer), "Error releasing g_min_buffer");
}

void gradient_descent(grid_t *g_in, grid_t *g_out, double dt, double alpha, double beta, double mass, int steps, v3d field, double T, double T_factor, gpu_t *gpu, bool use_gpu, int n_threads) {
    if (use_gpu)
	gradient_descent_gpu(g_in, g_out, dt, alpha, beta, mass, steps, field, T, T_factor, gpu);
    else if (n_threads > 1)
	gradient_descent_multiple(g_in, g_out, dt, alpha, beta, mass, steps, field, T, T_factor, n_threads);
    else
	gradient_descent_single(g_in, g_out, dt, alpha, beta, mass, steps, field, T, T_factor);
}
#endif
