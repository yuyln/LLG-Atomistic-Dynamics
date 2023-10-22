#include "./headers/constants.h"
#include "./headers/vec.h"
#include "./headers/random_extern.h"
#include "./headers/grid.h"
#include "./headers/funcs.h"

kernel void termal_step(global grid_t* g_out, global grid_t* g_old, double T, double qV1, double exp1, double exp2, int seed) {
    size_t I = get_global_id(0);
    v3d local_g_out = g_out->grid[I];
    v3d local_g_old = g_old->grid[I];
    pinning_t pin = g_out->pinning[I];

    tyche_state state;
    tyche_seed(&state, seed + I);

    double R = tyche_double(state);
    double delta = 1.0 / pow(1.0 + qV1 * R * R / pow(T, exp1), exp2);
    if (tyche_double(state) < 0.5)
        delta = -delta;
    
    local_g_out.x = local_g_old.x + delta;

    R = tyche_double(state);
    delta = 1.0 / pow(1.0 + qV1 * R * R / pow(T, exp1), exp2);
    if (tyche_double(state) < 0.5)
        delta = -delta;
    
    local_g_out.y = local_g_old.y + delta;

    R = tyche_double(state);
    delta = 1.0 / pow(1.0 + qV1 * R * R / pow(T, exp1), exp2);
    if (tyche_double(state) < 0.5)
        delta = -delta;
    
    local_g_out.z = local_g_old.z + delta;

    local_g_out = grid_normalize(local_g_out, pin);

    g_out->grid[I] = local_g_out;
}

kernel void hamiltonian_gpu(GLOBAL grid_t* g, GLOBAL double* ham_buffer, v3d field, double norm_time) {
    size_t I = get_global_id(0);
    int col = I % COLS;
    int row = (I - col) / COLS;

    grid_param_t gp = g->param;
    anisotropy_t ani = g->ani[I];
    region_param_t region = g->regions[I];
    v3d c = get_pbc_v3d(row, col, g->grid, ROWS, COLS, gp.pbc);
    v3d l = get_pbc_v3d(row, col - 1, g->grid, ROWS, COLS, gp.pbc);
    v3d r = get_pbc_v3d(row, col + 1,  g->grid, ROWS, COLS, gp.pbc);
    v3d u = get_pbc_v3d(row + 1, col, g->grid, ROWS, COLS, gp.pbc);
    v3d d = get_pbc_v3d(row - 1, col, g->grid, ROWS, COLS, gp.pbc);

    ham_buffer[I] = hamiltonian_I(row, col, c, l, r, u, d, gp, ani, region, field, norm_time);
}

kernel void reset_gpu(global grid_t* g_old, global grid_t* g_new) {
    size_t I = get_global_id(0);
    g_old->grid[I] = g_new->grid[I];
}

kernel void reset_v3d_gpu(global v3d *v1, global v3d *v2) {
    size_t I = get_global_id(0);
    v1[I] = v2[I];
}

kernel void step_gpu(global grid_t *g_old, global grid_t *g_new, v3d field, double dt, current_t cur, double norm_time) {
	size_t I = get_global_id(0);
    int col = I % COLS;
    int row = (I - col) / COLS;

    grid_param_t gp = g_old->param;
    anisotropy_t ani = g_old->ani[I];
    region_param_t region = g_old->regions[I];
    pinning_t pin = g_old->pinning[I];
    v3d c = get_pbc_v3d(row, col, g_old->grid, ROWS, COLS, gp.pbc);
    v3d l = get_pbc_v3d(row, col - 1, g_old->grid, ROWS, COLS, gp.pbc);
    v3d r = get_pbc_v3d(row, col + 1,  g_old->grid, ROWS, COLS, gp.pbc);
    v3d u = get_pbc_v3d(row + 1, col, g_old->grid, ROWS, COLS, gp.pbc);
    v3d d = get_pbc_v3d(row - 1, col, g_old->grid, ROWS, COLS, gp.pbc);

    v3d c_new = {0};

    v3d dm = grid_step(row, col, c, l, r, u, d, gp, region, ani, field, cur, dt, norm_time);

	c_new = v3d_add(c, dm);
    c_new = grid_normalize(c_new, pin);
    g_new->grid[I] = c_new;
}

kernel void process_data(global grid_t *g_old, global grid_t *g_new, v3d field, double dt, double norm_time, global info_pack_t *sim_info, int calc_energy) {
    size_t I = get_global_id(0);
    int col = I % COLS;
    int row = (I - col) / COLS;

    grid_param_t gp = g_old->param;
    anisotropy_t ani = g_old->ani[I];
    region_param_t region = g_old->regions[I];

    v3d c0 = get_pbc_v3d(row, col, g_old->grid, ROWS, COLS, gp.pbc);
    v3d c1 = get_pbc_v3d(row, col, g_new->grid, gp.rows, gp.cols, gp.pbc);
    v3d l1 = get_pbc_v3d(row, col - 1, g_new->grid, gp.rows, gp.cols, gp.pbc);
    v3d r1 = get_pbc_v3d(row, col + 1, g_new->grid, gp.rows, gp.cols, gp.pbc);
    v3d u1 = get_pbc_v3d(row + 1, col, g_new->grid, gp.rows, gp.cols, gp.pbc);
    v3d d1 = get_pbc_v3d(row - 1, col, g_new->grid, gp.rows, gp.cols, gp.pbc);

    uint64_t x = I % gp.cols;
    uint64_t y = (I - x) / gp.cols;

    double charge_i = charge(c1, l1, r1, u1, d1);
    double charge_i_old = charge_old(c1, l1, r1, u1, d1);
    sim_info[I].charge_cx = x * charge_i;
    sim_info[I].charge_cy = y * charge_i;

    v3d vt = velocity_weighted(c0, c1, c1, l1, r1, u1, d1, dt * 0.5);

    sim_info[I].vx = vt.x;
    sim_info[I].vy = vt.y;
    sim_info[I].avg_mag = c1;
    sim_info[I].charge_lattice = charge_i;
    sim_info[I].charge_finite = charge_i_old;
    if (calc_energy) {
        sim_info[I].energy = hamiltonian_I(row, col, c1, l1, r1, u1, d1, gp, ani, region, field, norm_time);
        sim_info[I].energy_exchange = 0.5 * exchange_energy(c1, l1, r1, u1, d1, gp, region);
        sim_info[I].energy_dm = 0.5 * dm_energy(c1, l1, r1, u1, d1, gp, region);
        sim_info[I].energy_zeeman = zeeman_energy(row, col, c1, gp, field, norm_time);
        sim_info[I].energy_anisotropy = anisotropy_energy(c1, ani);
        sim_info[I].energy_cubic_anisotropy = cubic_anisotropy_energy(c1, gp);
    }
}

kernel void gradient_step_gpu(global grid_t *g_aux, global v3d *g_p, global v3d *g_c, global v3d *g_n, double dt, double alpha, double beta, double mass, double T, global double *H, int seed, v3d field) {
    size_t j = get_global_id(0);
    int col = j % COLS;
    int row = (j - col) / COLS;

    grid_param_t gparam = g_aux->param;
    anisotropy_t ani = g_aux->ani[j];
    region_param_t region = g_aux->regions[j];
    pinning_t pin = g_aux->pinning[j];
    v3d c = get_pbc_v3d(row, col, g_c, ROWS, COLS, gparam.pbc);
    v3d l = get_pbc_v3d(row, col - 1, g_c, ROWS, COLS, gparam.pbc);
    v3d r = get_pbc_v3d(row, col + 1,  g_c, ROWS, COLS, gparam.pbc);
    v3d u = get_pbc_v3d(row + 1, col, g_c, ROWS, COLS, gparam.pbc);
    v3d d = get_pbc_v3d(row - 1, col, g_c, ROWS, COLS, gparam.pbc);

    v3d gp = g_p[j];
    v3d gn = g_n[j];
    v3d gc = g_c[j];

    v3d vel = gradient_descente_velocity(gp, gn, dt);
    v3d Heff = gradient_descent_force(row, col, c, l, r, u, d, gparam, region, ani, field, alpha, beta, vel);

    if (T != 0) {
        tyche_state state;
        tyche_seed(&state, seed + j);

        double R1 = 2.0 * tyche_double(state) - 1.0;
        double R2 = 2.0 * tyche_double(state) - 1.0;
        double R3 = 2.0 * tyche_double(state) - 1.0;

    
        Heff = v3d_add(Heff, v3d_scalar(v3d_c(R1, R2, R3), T));
    }

    gn = v3d_add(
   		 v3d_sub(v3d_scalar(gc, 2.0), gp),
   	             v3d_scalar(Heff, -dt * dt / mass)
    		);

    gn = grid_normalize(gn, pin);
    g_n[j] = gn;

    c = get_pbc_v3d(row, col, g_n, ROWS, COLS, gparam.pbc);
    l = get_pbc_v3d(row, col - 1, g_n, ROWS, COLS, gparam.pbc);
    r = get_pbc_v3d(row, col + 1,  g_n, ROWS, COLS, gparam.pbc);
    u = get_pbc_v3d(row + 1, col, g_n, ROWS, COLS, gparam.pbc);
    d = get_pbc_v3d(row - 1, col, g_n, ROWS, COLS, gparam.pbc);

    H[j] = hamiltonian_I(row, col, c, l, r, u, d, gparam, ani, region, field, 0.0);
}
