/**
@file
Implements a 512-bit tyche (Well-Equidistributed Long-period Linear) RNG.
S. Neves, F. Araujo, Fast and small nonlinear pseudorandom number generators for computer simulation, in: International Conference on Parallel Processing and Applied Mathematics, Springer, 2011, pp. 92–101.
*/
#define TYCHE_FLOAT_MULTI 5.4210108624275221700372640e-20f
#define TYCHE_DOUBLE_MULTI 5.4210108624275221700372640e-20

typedef union{
	struct{
		uint a,b,c,d;
	};
	ulong res;
} tyche_state;

#define TYCHE_ROT(a,b) (((a) << (b)) | ((a) >> (32 - (b))))

#define tyche_macro_ulong(state) (tyche_macro_advance(state), state.res)
#define tyche_macro_advance(state) ( \
	state.a += state.b, \
	state.d = TYCHE_ROT(state.d ^ state.a, 16), \
	state.c += state.d, \
	state.b = TYCHE_ROT(state.b ^ state.c, 12), \
	state.a += state.b, \
	state.d = TYCHE_ROT(state.d ^ state.a, 8), \
	state.c += state.d, \
	state.b = TYCHE_ROT(state.b ^ state.c, 7) \
)

#define tyche_ulong(state) (tyche_advance(&state), state.res)
void tyche_advance(tyche_state* state){
	state->a += state->b;
	state->d = TYCHE_ROT(state->d ^ state->a, 16);
	state->c += state->d;
	state->b = TYCHE_ROT(state->b ^ state->c, 12);
	state->a += state->b;
	state->d = TYCHE_ROT(state->d ^ state->a, 8);
	state->c += state->d;
	state->b = TYCHE_ROT(state->b ^ state->c, 7);
}
void tyche_seed(tyche_state* state, ulong seed){
	state->a = seed >> 32;
	state->b = seed;
	state->c = 2654435769;
	state->d = 1367130551 ^ (get_global_id(0) + get_global_size(0) * (get_global_id(1) + get_global_size(1) * get_global_id(2)));
	for(uint i=0;i<20;i++){
		tyche_advance(state);
	}
}

#define tyche_uint(state) ((uint)tyche_ulong(state))
#define tyche_float(state) (tyche_ulong(state)*TYCHE_FLOAT_MULTI)
#define tyche_double(state) (tyche_ulong(state)*TYCHE_DOUBLE_MULTI)
#define tyche_double2(state) tyche_double(state)
#include <grid.h>
#include <funcs.h>

kernel void TermalStep(global Grid* g_out, const global Grid* g_old, const double T, const double qV1, const double exp1, const double exp2, const int seed)
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

    GridNormalizeI(I, g_out);
}

kernel void HamiltonianGPU(global Grid* g, global double* ham_buffer, const Vec field)
{
    size_t I = get_global_id(0);
    ham_buffer[I] = HamiltonianI(I, g, field);
}

kernel void Reset(global Grid* g_old, global const Grid* g_new)
{
    size_t I = get_global_id(0);
    g_old->grid[I] = g_new->grid[I];
}

kernel void StepGPU(const global Grid *g_old, global Grid *g_new, Vec field, double dt, Current cur, double norm_time, int i, int cut, global Vec* vxvy_Qz_avg_mag)
{
	size_t I = get_global_id(0);
	g_new->grid[I] = VecAdd(g_old->grid[I], StepI(I, g_old, field, cur, dt, norm_time));
    GridNormalizeI(I, g_new);

	if (i % cut == 0)
	{
		vxvy_Qz_avg_mag[I].z = ChargeI(I, g_new->grid, g_old->param.rows, g_old->param.cols, g_old->param.lattice, g_old->param.lattice, g_old->param.pbc);
		Vec vt = VelWeightedI(I, g_new->grid, g_old->grid, g_new->grid, g_old->param.rows, g_old->param.cols, 
					g_old->param.lattice, g_old->param.lattice, 0.5 * dt * HBAR / fabs(g_old->param.exchange), g_old->param.pbc);
		vxvy_Qz_avg_mag[I].x = vt.x;
		vxvy_Qz_avg_mag[I].y = vt.y;
		vxvy_Qz_avg_mag[TOTAL + I] = g_new->grid[I];
	}
}