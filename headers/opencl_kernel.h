#ifndef __OPEN_CL_KERNEL
#define __OPEN_CL_KERNEL
/*static*/ const char kernel_data[] = "\
#include <random_extern.h>\n\
#include <grid.h>\n\
#include <funcs.h>\n\
\n\
kernel void TermalStep(global Grid* g_out, global Grid* g_old, double T, double qV1, double exp1, double exp2, int seed)\n\
{\n\
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
    GridNormalizeI(I, g_out->grid, g_out->pinning);\n\
}\n\
\n\
kernel void HamiltonianGPU(global Grid* g, global double* ham_buffer, Vec field)\n\
{\n\
    size_t I = get_global_id(0);\n\
    ham_buffer[I] = HamiltonianI(I, g->grid, &g->param, g->ani, g->regions, field);\n\
}\n\
\n\
kernel void Reset(global Grid* g_old, global Grid* g_new)\n\
{\n\
    size_t I = get_global_id(0);\n\
    g_old->grid[I] = g_new->grid[I];\n\
}\n\
\n\
kernel void ResetVec(global Vec *v1, global Vec *v2)\n\
{\n\
    size_t I = get_global_id(0);\n\
    v1[I] = v2[I];\n\
}\n\
\n\
kernel void StepGPU(global Grid *g_old, global Grid *g_new, Vec field, double dt, Current cur, double norm_time, int i, int cut, global Vec* vxvy_Ez_avg_mag_cp_ci, int calc_energy)\n\
{\n\
	size_t I = get_global_id(0);\n\
    Vec dMdt = StepI(I, g_old, field, cur, dt, norm_time);\n\
	g_new->grid[I] = VecAdd(g_old->grid[I], dMdt);\n\
    GridNormalizeI(I, g_new->grid, g_new->pinning);\n\
\n\
	if (i % cut == 0)\n\
	{\n\
		Vec vt = VelWeightedI(I, g_new->grid, g_old->grid, g_new->grid, g_old->param.rows, g_old->param.cols, \n\
					g_old->param.lattice, g_old->param.lattice, 0.5 * dt * HBAR / fabs(g_old->param.exchange), g_old->param.pbc);\n\
		vxvy_Ez_avg_mag_cp_ci[I].x = vt.x;\n\
		vxvy_Ez_avg_mag_cp_ci[I].y = vt.y;\n\
		vxvy_Ez_avg_mag_cp_ci[TOTAL + I] = g_new->grid[I];\n\
		vxvy_Ez_avg_mag_cp_ci[2 * TOTAL + I].x = ChargeI(I, g_new->grid, g_old->param.rows, g_old->param.cols, g_old->param.pbc);\n\
		vxvy_Ez_avg_mag_cp_ci[2 * TOTAL + I].y = ChargeI_old(I, g_new->grid, g_old->param.rows, g_old->param.cols, g_old->param.lattice, g_old->param.lattice, g_old->param.pbc);\n\
		if (calc_energy)\n\
			vxvy_Ez_avg_mag_cp_ci[I].z = HamiltonianI(I, g_new->grid, &g_new->param, g_new->ani, g_new->regions, field);\n\
	}\n\
}\n\
\n\
kernel void GradientStep(global Grid *g_aux, global Vec *g_p, global Vec *g_c, global Vec *g_n, double dt, double alpha, double beta, double mass, double T, global double *H, int seed, double J, Vec field)\n\
{\n\
    size_t j = get_global_id(0);\n\
\n\
\n\
    Vec vel = GradientDescentVelocity(g_p[j], g_n[j], dt);\n\
    Vec Heff = GradientDescentForce(j, g_aux, vel, g_c, field, J, alpha, beta);\n\
\n\
    if (T != 0)\n\
    {\n\
        tyche_state state;\n\
        tyche_seed(&state, seed + j);\n\
\n\
        double R1 = 2.0 * tyche_double(state) - 1.0;\n\
        double R2 = 2.0 * tyche_double(state) - 1.0;\n\
        double R3 = 2.0 * tyche_double(state) - 1.0;\n\
\n\
    \n\
        Heff = VecAdd(Heff, VecScalar(VecFrom(R1, R2, R3), T));\n\
    }\n\
\n\
    g_n[j] = VecAdd(\n\
   		    VecSub(VecScalar(g_c[j], 2.0), g_p[j]),\n\
   		    VecScalar(Heff, -dt * dt / mass)\n\
    		   );\n\
\n\
    GridNormalizeI(j, g_n, g_aux->pinning);\n\
    H[j] = HamiltonianI(j, g_n, &g_aux->param, g_aux->ani, g_aux->regions, field);\n\
}";
#endif