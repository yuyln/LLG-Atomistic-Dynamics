#ifndef __OPEN_CL_KERNEL
#define __OPEN_CL_KERNEL
/*static*/ const char kernel_data[] = "\
/*S. Neves, F. Araujo*/\n\
#define TYCHE_FLOAT_MULTI 5.4210108624275221700372640e-20f\n\
#define TYCHE_DOUBLE_MULTI 5.4210108624275221700372640e-20\n\
\n\
typedef union{\n\
	struct{\n\
		uint a,b,c,d;\n\
	};\n\
	ulong res;\n\
} tyche_state;\n\
\n\
#define TYCHE_ROT(a,b) (((a) << (b)) | ((a) >> (32 - (b))))\n\
#define tyche_macro_ulong(state) (tyche_macro_advance(state), state.res)\n\
#define tyche_macro_advance(state) ( \\n\
	state.a += state.b, \\n\
	state.d = TYCHE_ROT(state.d ^ state.a, 16), \\n\
	state.c += state.d, \\n\
	state.b = TYCHE_ROT(state.b ^ state.c, 12), \\n\
	state.a += state.b, \\n\
	state.d = TYCHE_ROT(state.d ^ state.a, 8), \\n\
	state.c += state.d, \\n\
	state.b = TYCHE_ROT(state.b ^ state.c, 7) \\n\
)\n\
\n\
#define tyche_ulong(state) (tyche_advance(&state), state.res)\n\
void tyche_advance(tyche_state* state){\n\
	state->a += state->b;\n\
	state->d = TYCHE_ROT(state->d ^ state->a, 16);\n\
	state->c += state->d;\n\
	state->b = TYCHE_ROT(state->b ^ state->c, 12);\n\
	state->a += state->b;\n\
	state->d = TYCHE_ROT(state->d ^ state->a, 8);\n\
	state->c += state->d;\n\
	state->b = TYCHE_ROT(state->b ^ state->c, 7);\n\
}\n\
void tyche_seed(tyche_state* state, ulong seed){\n\
	state->a = seed >> 32;\n\
	state->b = seed;\n\
	state->c = 2654435769;\n\
	state->d = 1367130551 ^ (get_global_id(0) + get_global_size(0) * (get_global_id(1) + get_global_size(1) * get_global_id(2)));\n\
	for(uint i=0;i<20;i++){\n\
		tyche_advance(state);\n\
	}\n\
}\n\
\n\
#define tyche_uint(state) ((uint)tyche_ulong(state))\n\
#define tyche_float(state) (tyche_ulong(state)*TYCHE_FLOAT_MULTI)\n\
#define tyche_double(state) (tyche_ulong(state)*TYCHE_DOUBLE_MULTI)\n\
#define tyche_double2(state) tyche_double(state)\n\
#include <grid.h>\n\
#include <funcs.h>\n\
\n\
kernel void TermalStep(global Grid* g_out, const global Grid* g_old, const double T, const double qV1, const double exp1, const double exp2, const int seed)\n\
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
    GridNormalizeI(I, g_out);\n\
}\n\
\n\
kernel void HamiltonianGPU(global Grid* g, global double* ham_buffer, const Vec field)\n\
{\n\
    size_t I = get_global_id(0);\n\
    ham_buffer[I] = HamiltonianI(I, g, field);\n\
}\n\
\n\
kernel void Reset(global Grid* g_old, global const Grid* g_new)\n\
{\n\
    size_t I = get_global_id(0);\n\
    g_old->grid[I] = g_new->grid[I];\n\
}\n\
\n\
kernel void StepGPU(const global Grid *g_old, global Grid *g_new, Vec field, double dt, Current cur, double norm_time, int i, int cut, global Vec* vxvy_Ez_avg_mag_cp_ci, int calc_energy)\n\
{\n\
	size_t I = get_global_id(0);\n\
	g_new->grid[I] = VecAdd(g_old->grid[I], StepI(I, g_old, field, cur, dt, norm_time));\n\
    GridNormalizeI(I, g_new);\n\
\n\
	if (i % cut == 0)\n\
	{\n\
		Vec vt = VelWeightedI(I, g_new->grid, g_old->grid, g_new->grid, g_old->param.rows, g_old->param.cols, \n\
					g_old->param.lattice, g_old->param.lattice, 0.5 * dt * HBAR / fabs(g_old->param.exchange), g_old->param.pbc);\n\
		vxvy_Ez_avg_mag_cp_ci[I].x = vt.x;\n\
		vxvy_Ez_avg_mag_cp_ci[I].y = vt.y;\n\
		vxvy_Ez_avg_mag_cp_ci[TOTAL + I] = g_new->grid[I];\n\
		vxvy_Ez_avg_mag_cp_ci[2 * TOTAL + I].x = ChargeI(I, g_new->grid, g_old->param.rows, g_old->param.cols, g_old->param.lattice, g_old->param.lattice, g_old->param.pbc);\n\
		vxvy_Ez_avg_mag_cp_ci[2 * TOTAL + I].y = ChargeI_old(I, g_new->grid, g_old->param.rows, g_old->param.cols, g_old->param.lattice, g_old->param.lattice, g_old->param.pbc);\n\
		if (calc_energy)\n\
			vxvy_Ez_avg_mag_cp_ci[I].z = HamiltonianI(I, g_new, field);\n\
	}\n\
}\n\
";
#endif