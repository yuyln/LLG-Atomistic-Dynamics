#include <random_extern.h>
#include <grid.h>
#include <funcs.h>

kernel void TermalStep(global Grid* g_out, global Grid* g_old, double T, double qV1, double exp1, double exp2, int seed)
{
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

    GridNormalizeI(I, g_out->grid, g_out->pinning);
}

kernel void HamiltonianGPU(global Grid* g, global double* ham_buffer, Vec field)
{
    size_t I = get_global_id(0);
    ham_buffer[I] = HamiltonianI(I, g->grid, &g->param, g->ani, g->regions, field);
}

kernel void Reset(global Grid* g_old, global Grid* g_new)
{
    size_t I = get_global_id(0);
    g_old->grid[I] = g_new->grid[I];
}

kernel void ResetVec(global Vec *v1, global Vec *v2)
{
    size_t I = get_global_id(0);
    v1[I] = v2[I];
}

kernel void StepGPU(global Grid *g_old, global Grid *g_new, Vec field, double dt, Current cur, double norm_time, int i, int cut, global Vec* vxvy_Ez_avg_mag_cp_ci, int calc_energy)
{
	size_t I = get_global_id(0);
	g_new->grid[I] = VecAdd(g_old->grid[I], StepI(I, g_old, field, cur, dt, norm_time));
    GridNormalizeI(I, g_new->grid, g_new->pinning);

	if (i % cut == 0)
	{
		Vec vt = VelWeightedI(I, g_new->grid, g_old->grid, g_new->grid, g_old->param.rows, g_old->param.cols, 
					g_old->param.lattice, g_old->param.lattice, 0.5 * dt * HBAR / fabs(g_old->param.exchange), g_old->param.pbc);
		vxvy_Ez_avg_mag_cp_ci[I].x = vt.x;
		vxvy_Ez_avg_mag_cp_ci[I].y = vt.y;
		vxvy_Ez_avg_mag_cp_ci[TOTAL + I] = g_new->grid[I];
		vxvy_Ez_avg_mag_cp_ci[2 * TOTAL + I].x = ChargeI(I, g_new->grid, g_old->param.rows, g_old->param.cols, g_old->param.pbc);
		vxvy_Ez_avg_mag_cp_ci[2 * TOTAL + I].y = ChargeI_old(I, g_new->grid, g_old->param.rows, g_old->param.cols, g_old->param.lattice, g_old->param.lattice, g_old->param.pbc);
		if (calc_energy)
			vxvy_Ez_avg_mag_cp_ci[I].z = HamiltonianI(I, g_new->grid, &g_new->param, g_new->ani, g_new->regions, field);
	}
}

kernel void GradientStep(global Grid *g_aux, global Vec *g_p, global Vec *g_c, global Vec *g_n, double dt, double alpha, double beta, double mass, double T, global double *H, int seed, double J, Vec field)
{
    size_t j = get_global_id(0);


    Vec vel = GradientDescentVelocity(g_p[j], g_n[j], dt);
    Vec Heff = GradientDescentForce(j, g_aux, vel, g_c, field, J, alpha, beta);

    if (T != 0)
    {
        tyche_state state;
        tyche_seed(&state, seed + j);

        double R1 = 2.0 * tyche_double(state) - 1.0;
        double R2 = 2.0 * tyche_double(state) - 1.0;
        double R3 = 2.0 * tyche_double(state) - 1.0;

    
        Heff = VecAdd(Heff, VecScalar(VecFrom(R1, R2, R3), T));
    }

    g_n[j] = VecAdd(
   		    VecSub(VecScalar(g_c[j], 2.0), g_p[j]),
   		    VecScalar(Heff, -dt * dt / mass)
    		   );

    GridNormalizeI(j, g_n, g_aux->pinning);
    H[j] = HamiltonianI(j, g_n, &g_aux->param, g_aux->ani, g_aux->regions, field);
}
