#ifndef __gsa
#define __gsa

#include "./headers/helpers.h"

void gsa(gsa_param_t param, grid_t* g_in, grid_t* g_out, v3d field) {
    grid_t g_min = grid_init_null(),
         g_old = grid_init_null(); {
        printf("qA: %e\n", param.qA);
        printf("qT: %e\n", param.qT);
        printf("qV: %e\n", param.qV);
        printf("T0: %e\n", param.T0);
        printf("outer_loop: %zu\n", param.outer_loop);
        printf("inner_loop: %zu\n", param.inner_loop);
    }

    grid_copy(&g_min, g_in);
    grid_copy(&g_old, g_in);
    grid_copy(g_out, g_in);

    double norm_factor = g_in->param.exchange * (g_in->param.exchange > 0? 1.0: -1.0);

    double H_old = hamiltonian(&g_old, field),
           H_new = hamiltonian(g_out, field),
           H_min = hamiltonian(&g_min, field);

    double qA1 = param.qA - 1.0,
           qV1 = param.qV - 1.0,
           qT1 = param.qT - 1.0,
           oneqA1 = 1.0 / qA1,
           exp1 = 2.0 / (3.0 - param.qV),
           exp2 = 1.0 / qV1 - 0.5;
    
    for (size_t outer = 1; outer <= param.outer_loop; ++outer) {
        srand(time(NULL));
        double t = 0.0,
               Tqt = param.T0 * (pow(2.0, qT1) - 1.0);
        for (size_t inner = 1; inner <= param.inner_loop; ++inner) {
            t += 1.0;
            double T = Tqt / (pow(t + 1.0, qT1) - 1.0);
            if (inner % (param.inner_loop / param.print_param) == 0)
                printf("outer: %zu inner: %zu H_min: %e T: %e\n", outer, inner, H_min, T);
            
            for (size_t I = 0; I < g_in->param.total; ++I) {
                double R = rand_double();
                double delta = 1.0 / pow(1.0 + qV1 * R * R / pow(T, exp1), exp2);
                if (rand_double() < 0.5)
                    delta = -delta;
                g_out->grid[I].x = g_old.grid[I].x + delta;

                //------------------------------

                R = rand_double();
                delta = 1.0 / pow(1.0 + qV1 * R * R / pow(T, exp1), exp2);
                if (rand_double() < 0.5)
                    delta = -delta;
                g_out->grid[I].y = g_old.grid[I].y + delta;

                //------------------------------

                R = rand_double();
                delta = 1.0 / pow(1.0 + qV1 * R * R / pow(T, exp1), exp2);
                if (rand_double() < 0.5)
                    delta = -delta;
                g_out->grid[I].z = g_old.grid[I].z + delta;

                grid_normalize(I, g_out->grid, g_out->pinning);
            }
            
            H_new = hamiltonian(g_out, field);

            if (H_new <= H_min) {
                H_min = H_new;
                copy_spins_to_allocated_grid(&g_min, g_out);   
            }

            if (H_new <= H_old) {
                H_old = H_new;
                copy_spins_to_allocated_grid(&g_old, g_out);
            }
            else {
                double df_norm = (H_new - H_old) / norm_factor;
                double pqa = 1.0 / pow(1.0 + qA1 * df_norm / T, oneqA1);
                if (rand_double() < pqa) {
                    H_old = H_new;
                    copy_spins_to_allocated_grid(&g_old, g_out);
                }
            }
        }
    }
    grid_copy(g_out, &g_min);
    grid_free(&g_min);
    grid_free(&g_old);
}

void gsagpu_t(gsa_param_t param, grid_t* g_in, grid_t* g_out, v3d field, gpu_t *gpu) {
    grid_t g_min = grid_init_null(),
         g_old = grid_init_null(); {
        printf("qA: %e\n", param.qA);
        printf("qT: %e\n", param.qT);
        printf("qV: %e\n", param.qV);
        printf("T0: %e\n", param.T0);
        printf("outer_loop: %zu\n", param.outer_loop);
        printf("inner_loop: %zu\n", param.inner_loop);
    }

    grid_copy(&g_min, g_in);
    grid_copy(&g_old, g_in);
    grid_copy(g_out, g_in);

    double norm_factor = g_in->param.exchange * (g_in->param.exchange > 0? 1.0: -1.0);

    double H_old = hamiltonian(&g_old, field),
           H_new = hamiltonian(g_out, field),
           H_min = hamiltonian(&g_min, field);

    grid_free(&g_min);
    grid_free(&g_old);

    double qA1 = param.qA - 1.0,
           qV1 = param.qV - 1.0,
           qT1 = param.qT - 1.0,
           oneqA1 = 1.0 / qA1,
           exp1 = 2.0 / (3.0 - param.qV),
           exp2 = 1.0 / qV1 - 0.5;
    
    cl_mem g_min_buffer = clw_create_buffer(find_grid_size_bytes(g_in), gpu->ctx, CL_MEM_READ_WRITE), 
           g_old_buffer = clw_create_buffer(find_grid_size_bytes(g_in), gpu->ctx, CL_MEM_READ_WRITE),
           g_out_buffer = clw_create_buffer(find_grid_size_bytes(g_in), gpu->ctx, CL_MEM_READ_WRITE),
           ham_buffer = clw_create_buffer(sizeof(double) * g_in->param.total, gpu->ctx, CL_MEM_READ_WRITE);
    
    full_grid_write_buffer(gpu->queue, g_min_buffer, g_in);
    full_grid_write_buffer(gpu->queue, g_old_buffer, g_in);
    full_grid_write_buffer(gpu->queue, g_out_buffer, g_in);

    double* ham_gpu = (double*)calloc(g_in->param.total, sizeof(double));


    size_t global = g_in->param.total,
           local = gcd(global, 32);

    clw_set_kernel_arg(gpu->kernels[0], 0, sizeof(cl_mem), &g_out_buffer);
    clw_set_kernel_arg(gpu->kernels[0], 1, sizeof(cl_mem), &g_old_buffer);
    clw_set_kernel_arg(gpu->kernels[0], 3, sizeof(double), &qV1);
    clw_set_kernel_arg(gpu->kernels[0], 4, sizeof(double), &exp1);
    clw_set_kernel_arg(gpu->kernels[0], 5, sizeof(double), &exp2);
    
    clw_set_kernel_arg(gpu->kernels[1], 0, sizeof(cl_mem), &g_out_buffer);
    clw_set_kernel_arg(gpu->kernels[1], 1, sizeof(cl_mem), &ham_buffer);
    clw_set_kernel_arg(gpu->kernels[1], 2, sizeof(v3d) ,&field);

    clw_set_kernel_arg(gpu->kernels[2], 1, sizeof(cl_mem), &g_out_buffer);

    for (size_t outer = 1; outer <= param.outer_loop; ++outer) {
        srand(time(NULL));
        double t = 0.0,
               Tqt = param.T0 * (pow(2.0, qT1) - 1.0);
        for (size_t inner = 1; inner <= param.inner_loop; ++inner) {
            t += 1.0;
            double T = Tqt / (pow(t + 1.0, qT1) - 1.0);
            clw_set_kernel_arg(gpu->kernels[0], 2, sizeof(double) ,&T);
            if (inner % (param.inner_loop / param.print_param) == 0)
                printf("outer: %zu inner: %zu H_min: %e T: %e\n", outer, inner, H_min, T);
            int seed = rand();
            clw_set_kernel_arg(gpu->kernels[0], 6, sizeof(int) ,&seed);
            clw_enqueue_nd(gpu->queue, gpu->kernels[0], 1, NULL, &global, &local);
            //clw_finish(gpu->queue);

            clw_enqueue_nd(gpu->queue, gpu->kernels[1], 1, NULL, &global, &local);
            //clw_finish(gpu->queue);

            clw_read_buffer(ham_buffer, ham_gpu, sizeof(double) * g_in->param.total, 0, gpu->queue);

            double HH = 0.0;
            for (size_t i = 0; i < g_in->param.total; ++i)
                HH += ham_gpu[i];
            
            H_new = HH;
            
            if (H_new <= H_min) {
                H_min = H_new;
                clw_set_kernel_arg(gpu->kernels[2], 0, sizeof(cl_mem), &g_min_buffer);
                clw_enqueue_nd(gpu->queue, gpu->kernels[2], 1, NULL, &global, &local);
                //clw_finish(gpu->queue);
            }

            if (H_new <= H_old) {
                H_old = H_new;
                clw_set_kernel_arg(gpu->kernels[2], 0, sizeof(cl_mem), &g_old_buffer);
                clw_enqueue_nd(gpu->queue, gpu->kernels[2], 1, NULL, &global, &local);
                //clw_finish(gpu->queue);
            }
            else {
                double df_norm = (H_new - H_old) / norm_factor;
                double pqa = 1.0 / pow(1.0 + qA1 * df_norm / T, oneqA1);
                if (rand_double() < pqa) {
                    H_old = H_new;
                    clw_set_kernel_arg(gpu->kernels[2], 0, sizeof(cl_mem), &g_old_buffer);
                    clw_enqueue_nd(gpu->queue, gpu->kernels[2], 1, NULL, &global, &local);
                    //clw_finish(gpu->queue);
                }
            }
        }
    }
    read_full_grid_buffer(gpu->queue, g_min_buffer, g_out);

    free(ham_gpu);
    clw_print_cl_error(stderr, clReleaseMemObject(g_min_buffer), "Could not release g_min_buffer");
    clw_print_cl_error(stderr, clReleaseMemObject(g_out_buffer), "Could not release g_out_buffer");
    clw_print_cl_error(stderr, clReleaseMemObject(g_old_buffer), "Could not release g_old_buffer");
    clw_print_cl_error(stderr, clReleaseMemObject(ham_buffer), "Could not release ham_buffer");
}
#endif
