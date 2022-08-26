#ifndef __OPEN_CL_KERNEL
#define __OPEN_CL_KERNEL
static const char kernel_data[] = "\
/**\n\
@file\n\
\n\
Implements a 512-bit tyche (Well-Equidistributed Long-period Linear) RNG.\n\
\n\
S. Neves, F. Araujo, Fast and small nonlinear pseudorandom number generators for computer simulation, in: International Conference on Parallel Processing and Applied Mathematics, Springer, 2011, pp. 92–101.\n\
*/\n\
#define TYCHE_FLOAT_MULTI 5.4210108624275221700372640e-20f\n\
#define TYCHE_DOUBLE_MULTI 5.4210108624275221700372640e-20\n\
\n\
/**\n\
State of tyche RNG.\n\
*/\n\
typedef union{\n\
	struct{\n\
		uint a,b,c,d;\n\
	};\n\
	ulong res;\n\
} tyche_state;\n\
\n\
#define TYCHE_ROT(a,b) (((a) << (b)) | ((a) >> (32 - (b))))\n\
\n\
/**\n\
Generates a random 64-bit unsigned integer using tyche RNG.\n\
\n\
This is alternative, macro implementation of tyche RNG.\n\
\n\
@param state State of the RNG to use.\n\
*/\n\
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
/**\n\
Generates a random 64-bit unsigned integer using tyche RNG.\n\
\n\
@param state State of the RNG to use.\n\
*/\n\
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
\n\
/**\n\
Seeds tyche RNG.\n\
\n\
@param state Variable, that holds state of the generator to be seeded.\n\
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).\n\
*/\n\
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
/**\n\
Generates a random 32-bit unsigned integer using tyche RNG.\n\
\n\
@param state State of the RNG to use.\n\
*/\n\
#define tyche_uint(state) ((uint)tyche_ulong(state))\n\
\n\
/**\n\
Generates a random float using tyche RNG.\n\
\n\
@param state State of the RNG to use.\n\
*/\n\
#define tyche_float(state) (tyche_ulong(state)*TYCHE_FLOAT_MULTI)\n\
\n\
/**\n\
Generates a random double using tyche RNG.\n\
\n\
@param state State of the RNG to use.\n\
*/\n\
#define tyche_double(state) (tyche_ulong(state)*TYCHE_DOUBLE_MULTI)\n\
\n\
/**\n\
Generates a random double using tyche RNG. Since tyche returns 64-bit numbers this is equivalent to tyche_double.\n\
\n\
@param state State of the RNG to use.\n\
*/\n\
#define tyche_double2(state) tyche_double(state)\n\
\n\
\n\
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
    g_out->grid[I].x = g_old->grid[I].x + delta;\n\
\n\
    R = tyche_double(state);\n\
    delta = 1.0 / pow(1.0 + qV1 * R * R / pow(T, exp1), exp2);\n\
    if (tyche_double(state) < 0.5)\n\
        delta = -delta;\n\
    \n\
    g_out->grid[I].z = g_old->grid[I].z + delta;\n\
\n\
    GridNormalizeI(g_out, I);\n\
}\n\
\n\
kernel void HamiltonianGPU(global const Grid* g, global double* ham_buffer, const Vec field)\n\
{\n\
    size_t I = get_global_id(0);\n\
    ham_buffer[I] = HamiltonianI(I, g, field);\n\
}";
#endif