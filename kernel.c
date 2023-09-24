#include "./headers/constants.h"
#include "./headers/vec.h"
#include "./headers/random_extern.h"
#include "./headers/grid.h"
#include "./headers/funcs.h"

kernel void termal_step(global grid_t* g_out, global grid_t* g_old, double T, double qV1, double exp1, double exp2, int seed) {
    size_t I = get_global_id(0);

    tyche_state state;
    tyche_seed(&state, seed + I);

    double R = tyche_double(state);
    double delta = 1.0 / pow(1.0 + qV1 * R * R / pow(T, exp1), exp2);
    if (tyche_double(state) < 0.5)
        delta = -delta;
    
    g_out->grid[I].x = g_old->grid[I].x + delta;

    R = tyche_double(state);
    delta = 1.0 / pow(1.0 + qV1 * R * R / pow(T, exp1), exp2);
    if (tyche_double(state) < 0.5)
        delta = -delta;
    
    g_out->grid[I].y = g_old->grid[I].y + delta;

    R = tyche_double(state);
    delta = 1.0 / pow(1.0 + qV1 * R * R / pow(T, exp1), exp2);
    if (tyche_double(state) < 0.5)
        delta = -delta;
    
    g_out->grid[I].z = g_old->grid[I].z + delta;

    grid_normalize(I, g_out->grid, g_out->pinning);
}

kernel void hamiltonian_gpu(global grid_t* g, global double* ham_buffer, v3d field) {
    size_t I = get_global_id(0);
    ham_buffer[I] = hamiltonian_I(I, g->grid, &g->param, g->ani, g->regions, field);
}

kernel void reset_gpu(global grid_t* g_old, global grid_t* g_new) {
    size_t I = get_global_id(0);
    g_old->grid[I] = g_new->grid[I];
}

kernel void reset_v3d_gpu(global v3d *v1, global v3d *v2) {
    size_t I = get_global_id(0);
    v1[I] = v2[I];
}

kernel void step_gpu(global grid_t *g_old, global grid_t *g_new, v3d field, double dt, current_t cur, double norm_time, int i, int cut, global v3d* vxvy_Ez_avg_mag_cp_ci, int calc_energy) {
	size_t I = get_global_id(0);
    v3d dMdt = step(I, g_old, field, cur, dt, norm_time);
	g_new->grid[I] = v3d_add(g_old->grid[I], dMdt);
    grid_normalize(I, g_new->grid, g_new->pinning);

	if (i % cut == 0) {
		v3d vt = velocity_weighted(I, g_new->grid, g_old->grid, g_new->grid, g_old->param.rows, g_old->param.cols, 
					g_old->param.lattice, g_old->param.lattice, 0.5 * dt * HBAR / fabs(g_old->param.exchange), g_old->param.pbc);
		vxvy_Ez_avg_mag_cp_ci[I].x = vt.x;
		vxvy_Ez_avg_mag_cp_ci[I].y = vt.y;
		vxvy_Ez_avg_mag_cp_ci[TOTAL + I] = g_new->grid[I];
		vxvy_Ez_avg_mag_cp_ci[2 * TOTAL + I].x = charge(I, g_new->grid, g_old->param.rows, g_old->param.cols, g_old->param.pbc);
		vxvy_Ez_avg_mag_cp_ci[2 * TOTAL + I].y = charge_old(I, g_new->grid, g_old->param.rows, g_old->param.cols, g_old->param.lattice, g_old->param.lattice, g_old->param.pbc);
		if (calc_energy)
			vxvy_Ez_avg_mag_cp_ci[I].z = hamiltonian_I(I, g_new->grid, &g_new->param, g_new->ani, g_new->regions, field);
	}
}

kernel void gradient_step_gpu(global grid_t *g_aux, global v3d *g_p, global v3d *g_c, global v3d *g_n, double dt, double alpha, double beta, double mass, double T, global double *H, int seed, double J, v3d field) {
    size_t j = get_global_id(0);


    v3d vel = gradient_descente_velocity(g_p[j], g_n[j], dt);
    v3d Heff = gradient_descent_force(j, g_aux, vel, g_c, field, J, alpha, beta);

    if (T != 0) {
        tyche_state state;
        tyche_seed(&state, seed + j);

        double R1 = 2.0 * tyche_double(state) - 1.0;
        double R2 = 2.0 * tyche_double(state) - 1.0;
        double R3 = 2.0 * tyche_double(state) - 1.0;

    
        Heff = v3d_add(Heff, v3d_scalar(v3d_c(R1, R2, R3), T));
    }

    g_n[j] = v3d_add(
   		    v3d_sub(v3d_scalar(g_c[j], 2.0), g_p[j]),
   		    v3d_scalar(Heff, -dt * dt / mass)
    		   );

    grid_normalize(j, g_n, g_aux->pinning);
    H[j] = hamiltonian_I(j, g_n, &g_aux->param, g_aux->ani, g_aux->regions, field);
}
