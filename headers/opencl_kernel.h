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
    v3d local_g_out = g_out->grid[I];\n\
    v3d local_g_old = g_old->grid[I];\n\
    pinning_t pin = g_out->pinning[I];\n\
\n\
    tyche_state state;\n\
    tyche_seed(&state, seed + I);\n\
\n\
    double R = tyche_double(state);\n\
    double delta = 1.0 / pow(1.0 + qV1 * R * R / pow(T, exp1), exp2);\n\
    if (tyche_double(state) < 0.5)\n\
        delta = -delta;\n\
    \n\
    local_g_out.x = local_g_old.x + delta;\n\
\n\
    R = tyche_double(state);\n\
    delta = 1.0 / pow(1.0 + qV1 * R * R / pow(T, exp1), exp2);\n\
    if (tyche_double(state) < 0.5)\n\
        delta = -delta;\n\
    \n\
    local_g_out.y = local_g_old.y + delta;\n\
\n\
    R = tyche_double(state);\n\
    delta = 1.0 / pow(1.0 + qV1 * R * R / pow(T, exp1), exp2);\n\
    if (tyche_double(state) < 0.5)\n\
        delta = -delta;\n\
    \n\
    local_g_out.z = local_g_old.z + delta;\n\
\n\
    local_g_out = grid_normalize(local_g_out, pin);\n\
\n\
    g_out->grid[I] = local_g_out;\n\
}\n\
\n\
kernel void hamiltonian_gpu(GLOBAL grid_t* g, GLOBAL double* ham_buffer, v3d field) {\n\
    size_t I = get_global_id(0);\n\
    int col = I % COLS;\n\
    int row = (I - col) / COLS;\n\
\n\
    grid_param_t gp = g->param;\n\
    anisotropy_t ani = g->ani[I];\n\
    region_param_t region = g->regions[I];\n\
    v3d c = get_pbc_v3d(row, col, g->grid, ROWS, COLS, gp.pbc);\n\
    v3d l = get_pbc_v3d(row, col - 1, g->grid, ROWS, COLS, gp.pbc);\n\
    v3d r = get_pbc_v3d(row, col + 1,  g->grid, ROWS, COLS, gp.pbc);\n\
    v3d u = get_pbc_v3d(row + 1, col, g->grid, ROWS, COLS, gp.pbc);\n\
    v3d d = get_pbc_v3d(row - 1, col, g->grid, ROWS, COLS, gp.pbc);\n\
\n\
    ham_buffer[I] = hamiltonian_I(row, col, c, l, r, u, d, gp, ani, region, field);\n\
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
    int col = I % COLS;\n\
    int row = (I - col) / COLS;\n\
\n\
    grid_param_t gp = g_old->param;\n\
    anisotropy_t ani = g_old->ani[I];\n\
    region_param_t region = g_old->regions[I];\n\
    pinning_t pin = g_old->pinning[I];\n\
    v3d c = get_pbc_v3d(row, col, g_old->grid, ROWS, COLS, gp.pbc);\n\
    v3d l = get_pbc_v3d(row, col - 1, g_old->grid, ROWS, COLS, gp.pbc);\n\
    v3d r = get_pbc_v3d(row, col + 1,  g_old->grid, ROWS, COLS, gp.pbc);\n\
    v3d u = get_pbc_v3d(row + 1, col, g_old->grid, ROWS, COLS, gp.pbc);\n\
    v3d d = get_pbc_v3d(row - 1, col, g_old->grid, ROWS, COLS, gp.pbc);\n\
\n\
    v3d c_old = c;\n\
    v3d c_new = {0};\n\
\n\
    v3d dm = grid_step(row, col, c, l, r, u, d, gp, region, ani, field, cur, dt, norm_time);\n\
\n\
	c_new = v3d_add(c_old, dm);\n\
    c_new = grid_normalize(c_new, pin);\n\
    g_new->grid[I] = c_new;\n\
\n\
	if (i % cut == 0) {\n\
		v3d vt = velocity_weighted(c_old, c_old, c_new, l, r, u, d, gp.lattice, gp.lattice, 2.0 * dt * HBAR / fabs(gp.exchange));\n\
		vxvy_Ez_avg_mag_cp_ci[I].x = vt.x;\n\
		vxvy_Ez_avg_mag_cp_ci[I].y = vt.y;\n\
		vxvy_Ez_avg_mag_cp_ci[TOTAL + I] = g_new->grid[I];\n\
		vxvy_Ez_avg_mag_cp_ci[2 * TOTAL + I].x = charge(c, l, r, u, d);\n\
		vxvy_Ez_avg_mag_cp_ci[2 * TOTAL + I].y = charge_old(c, l, r, u, d, gp.lattice, gp.lattice);\n\
		vxvy_Ez_avg_mag_cp_ci[I].z = hamiltonian_I(row, col, c, l, r, u, d, gp, ani, region, field);\n\
	}\n\
}\n\
\n\
kernel void gradient_step_gpu(global grid_t *g_aux, global v3d *g_p, global v3d *g_c, global v3d *g_n, double dt, double alpha, double beta, double mass, double T, global double *H, int seed, v3d field) {\n\
    size_t j = get_global_id(0);\n\
    int col = j % COLS;\n\
    int row = (j - col) / COLS;\n\
\n\
    grid_param_t gparam = g_aux->param;\n\
    anisotropy_t ani = g_aux->ani[j];\n\
    region_param_t region = g_aux->regions[j];\n\
    pinning_t pin = g_aux->pinning[j];\n\
    v3d c = get_pbc_v3d(row, col, g_c, ROWS, COLS, gparam.pbc);\n\
    v3d l = get_pbc_v3d(row, col - 1, g_c, ROWS, COLS, gparam.pbc);\n\
    v3d r = get_pbc_v3d(row, col + 1,  g_c, ROWS, COLS, gparam.pbc);\n\
    v3d u = get_pbc_v3d(row + 1, col, g_c, ROWS, COLS, gparam.pbc);\n\
    v3d d = get_pbc_v3d(row - 1, col, g_c, ROWS, COLS, gparam.pbc);\n\
\n\
    v3d gp = g_p[j];\n\
    v3d gn = g_n[j];\n\
    v3d gc = g_c[j];\n\
\n\
    v3d vel = gradient_descente_velocity(gp, gn, dt);\n\
    v3d Heff = gradient_descent_force(row, col, c, l, r, u, d, gparam, region, ani, field, alpha, beta, vel);\n\
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
    gn = v3d_add(\n\
   		 v3d_sub(v3d_scalar(gc, 2.0), gp),\n\
   	             v3d_scalar(Heff, -dt * dt / mass)\n\
    		);\n\
\n\
    gn = grid_normalize(gn, pin);\n\
    g_n[j] = gn;\n\
\n\
    c = get_pbc_v3d(row, col, g_n, ROWS, COLS, gparam.pbc);\n\
    l = get_pbc_v3d(row, col - 1, g_n, ROWS, COLS, gparam.pbc);\n\
    r = get_pbc_v3d(row, col + 1,  g_n, ROWS, COLS, gparam.pbc);\n\
    u = get_pbc_v3d(row + 1, col, g_n, ROWS, COLS, gparam.pbc);\n\
    d = get_pbc_v3d(row - 1, col, g_n, ROWS, COLS, gparam.pbc);\n\
\n\
    H[j] = hamiltonian_I(row, col, c, l, r, u, d, gparam, ani, region, field);\n\
}\n\
";
#endif