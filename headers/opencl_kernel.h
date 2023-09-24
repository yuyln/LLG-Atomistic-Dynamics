#ifndef __OPEN_CL_KERNEL
#define __OPEN_CL_KERNEL
/*static*/ const char kernel_data[] = "\
#include \"./headers/constants.h\"\n\
#include \"./headers/vec.h\"\n\
#include \"./headers/random_extern.h\"\n\
#include \"./headers/grid.h\"\n\
#include \"./headers/funcs.h\"\n\
\n\
kernel void termal_step(global grid_t* g_out, global grid_t* g_old, double T, double qV1, double exp1, double exp2, int seed) {\n\
    size_t I = get_global_id(0);\n\
\n\
    tyche_state state;\n\
    tyche_seed(&state, seed + I);\n\
\n\
    double R = tyche_double(state);\n\
    double delta = 1.0 / pow(1.0 + qV1 * R * R / pow(T, exp1), exp2);\n\
    if (tyche_double(state) < 0.5)\n\
        delta = -delta;\n\
    \n\
    g_out->grid[I].x = g_old->grid[I].x + delta;\n\
\n\
    R = tyche_double(state);\n\
    delta = 1.0 / pow(1.0 + qV1 * R * R / pow(T, exp1), exp2);\n\
    if (tyche_double(state) < 0.5)\n\
        delta = -delta;\n\
    \n\
    g_out->grid[I].y = g_old->grid[I].y + delta;\n\
\n\
    R = tyche_double(state);\n\
    delta = 1.0 / pow(1.0 + qV1 * R * R / pow(T, exp1), exp2);\n\
    if (tyche_double(state) < 0.5)\n\
        delta = -delta;\n\
    \n\
    g_out->grid[I].z = g_old->grid[I].z + delta;\n\
\n\
    grid_normalize(I, g_out->grid, g_out->pinning);\n\
}\n\
\n\
kernel void hamiltonian_gpu(global grid_t* g, global double* ham_buffer, v3d field) {\n\
    size_t I = get_global_id(0);\n\
    ham_buffer[I] = hamiltonian_I(I, g->grid, &g->param, g->ani, g->regions, field);\n\
}\n\
\n\
kernel void reset_gpu(global grid_t* g_old, global grid_t* g_new) {\n\
    size_t I = get_global_id(0);\n\
    g_old->grid[I] = g_new->grid[I];\n\
}\n\
\n\
kernel void reset_v3d_gpu(global v3d *v1, global v3d *v2) {\n\
    size_t I = get_global_id(0);\n\
    v1[I] = v2[I];\n\
}\n\
\n\
kernel void step_gpu(global grid_t *g_old, global grid_t *g_new, v3d field, double dt, current_t cur, double norm_time, int i, int cut, global v3d* vxvy_Ez_avg_mag_cp_ci, int calc_energy) {\n\
	size_t I = get_global_id(0);\n\
    v3d dMdt = step(I, g_old, field, cur, dt, norm_time);\n\
	g_new->grid[I] = v3d_add(g_old->grid[I], dMdt);\n\
    grid_normalize(I, g_new->grid, g_new->pinning);\n\
\n\
	if (i % cut == 0) {\n\
		v3d vt = velocity_weighted(I, g_new->grid, g_old->grid, g_new->grid, g_old->param.rows, g_old->param.cols, \n\
					g_old->param.lattice, g_old->param.lattice, 0.5 * dt * HBAR / fabs(g_old->param.exchange), g_old->param.pbc);\n\
		vxvy_Ez_avg_mag_cp_ci[I].x = vt.x;\n\
		vxvy_Ez_avg_mag_cp_ci[I].y = vt.y;\n\
		vxvy_Ez_avg_mag_cp_ci[TOTAL + I] = g_new->grid[I];\n\
		vxvy_Ez_avg_mag_cp_ci[2 * TOTAL + I].x = charge(I, g_new->grid, g_old->param.rows, g_old->param.cols, g_old->param.pbc);\n\
		vxvy_Ez_avg_mag_cp_ci[2 * TOTAL + I].y = charge_old(I, g_new->grid, g_old->param.rows, g_old->param.cols, g_old->param.lattice, g_old->param.lattice, g_old->param.pbc);\n\
		if (calc_energy)\n\
			vxvy_Ez_avg_mag_cp_ci[I].z = hamiltonian_I(I, g_new->grid, &g_new->param, g_new->ani, g_new->regions, field);\n\
	}\n\
}\n\
\n\
kernel void gradient_step_gpu(global grid_t *g_aux, global v3d *g_p, global v3d *g_c, global v3d *g_n, double dt, double alpha, double beta, double mass, double T, global double *H, int seed, double J, v3d field) {\n\
    size_t j = get_global_id(0);\n\
\n\
\n\
    v3d vel = gradient_descente_velocity(g_p[j], g_n[j], dt);\n\
    v3d Heff = gradient_descent_force(j, g_aux, vel, g_c, field, J, alpha, beta);\n\
\n\
    if (T != 0) {\n\
        tyche_state state;\n\
        tyche_seed(&state, seed + j);\n\
\n\
        double R1 = 2.0 * tyche_double(state) - 1.0;\n\
        double R2 = 2.0 * tyche_double(state) - 1.0;\n\
        double R3 = 2.0 * tyche_double(state) - 1.0;\n\
\n\
    \n\
        Heff = v3d_add(Heff, v3d_scalar(v3d_c(R1, R2, R3), T));\n\
    }\n\
\n\
    g_n[j] = v3d_add(\n\
   		    v3d_sub(v3d_scalar(g_c[j], 2.0), g_p[j]),\n\
   		    v3d_scalar(Heff, -dt * dt / mass)\n\
    		   );\n\
\n\
    grid_normalize(j, g_n, g_aux->pinning);\n\
    H[j] = hamiltonian_I(j, g_n, &g_aux->param, g_aux->ani, g_aux->regions, field);\n\
}\n\
";
#endif