#include "constants.h"
#include "grid_types.h"
#include "simulation_funcs.h"

kernel void gpu_step(GLOBAL grid_site_params *gs, GLOBAL v3d *input, GLOBAL v3d *out, double dt, double time, grid_info gi) {
    size_t id = get_global_id(0);

    if (id >= (gi.rows * gi.cols * gi.depth))
        return;

    int k = id / (gi.rows * gi.cols);
    int col = id % gi.cols;
    int row = id / gi.cols;

    parameters param = (parameters){};
    param.rows = gi.rows;
    param.cols = gi.cols;
    param.depth = gi.depth;
    param.gs = gs[id];
    param.lattice = gi.lattice;
    param.m = apply_pbc(input, gi.pbc, row, col, k, gi.rows, gi.cols, gi.depth);
    param.neigh.left = apply_pbc(input, gi.pbc, row, col - 1, k, gi.rows, gi.cols, gi.depth);
    param.neigh.right = apply_pbc(input, gi.pbc, row, col + 1, k, gi.rows, gi.cols, gi.depth);
    param.neigh.up = apply_pbc(input, gi.pbc, row + 1, col, k, gi.rows, gi.cols, gi.depth);
    param.neigh.down = apply_pbc(input, gi.pbc, row - 1, col, k, gi.rows, gi.cols, gi.depth);
    param.neigh.front = apply_pbc(input, gi.pbc, row, col, k + 1, gi.rows, gi.cols, gi.depth);
    param.neigh.back = apply_pbc(input, gi.pbc, row, col, k - 1, gi.rows, gi.cols, gi.depth);
    param.time = time;
    tyche_i_state state;
    int seed = *((int*)(&time));
    seed = seed << 16;
    tyche_i_seed(&state, seed + id);
    param.state = &state;

#ifdef INCLUDE_DIPOLAR
    param.dipolar_field = v3d_s(0.0);
    for (int dr = -param.rows / 2; dr < param.rows / 2; ++dr) {
        double dy = dr * gi.lattice;
        for (int dc = -param.cols / 2; dc < param.cols / 2; ++dc) {
            double dx = dc * gi.lattice;
            for (int dk = -param.depth / 2; dk < param.depth / 2; ++ dk) {
                if (dr == 0 && dc == 0 && dk == 0)
                    continue;
                double dz = dk * gi.lattice;
                double r_ij_ = sqrt(dx * dx + dy * dy + dz * dz);
                v3d r_ij = v3d_normalize(v3d_c(dx, dy, dz));
                v3d mj;
                grid_site_params gj;
                apply_pbc_complete(gs, input, &mj, &gj, gi.pbc, row + dr, col + dc, k + dk, param.rows, param.cols, param.depth);
                double factor = -MU_0 * param.gs.mu * gj.mu / (4.0 * M_PI);
                v3d interaction = v3d_scalar(r_ij, 3.0 * v3d_dot(mj, r_ij));
                interaction = v3d_sub(interaction, mj);
                interaction = v3d_scalar(interaction, factor / (r_ij_ * r_ij_ * r_ij_));
                param.dipolar_field = v3d_sum(param.dipolar_field, interaction);
            }
        }
    }
#endif

    double temperature = generate_temperature(param.gs, param.time);
    if (!CLOSE_ENOUGH(temperature, 0.0, EPS)) {
        param.temperature_effect = v3d_scalar(v3d_normalize(v3d_c(normal_distribution(param.state), normal_distribution(param.state), normal_distribution(param.state))),
                sqrt(2.0 * param.gs.alpha * KB * temperature / (param.gs.gamma * param.gs.mu * dt)));
    }

    out[id] = v3d_normalize(param.gs.pin.pinned? param.gs.pin.dir: v3d_sum(param.m, step_llg(param, dt)));
}

kernel void extract_info(GLOBAL grid_site_params *gs, GLOBAL v3d *m0, GLOBAL v3d *m1, GLOBAL information_packed *info, double dt, double time, grid_info gi) {
    size_t id = get_global_id(0);

    if (id >= (gi.rows * gi.cols * gi.depth))
        return;

    int col = id % gi.cols;
    int row = id / gi.cols;
    int k = id / (gi.rows * gi.cols);

    parameters param;
    param.rows = gi.rows;
    param.cols = gi.cols;
    param.depth = gi.depth;
    param.gs = gs[id];
    param.m = m0[id];
    param.lattice = gi.lattice;
    v3d dm = v3d_sub(m1[id], param.m);
    param.m = apply_pbc(m0, gi.pbc, row, col, k, gi.rows, gi.cols, gi.depth);
    param.neigh.left = apply_pbc(m0, gi.pbc, row, col - 1, k, gi.rows, gi.cols, gi.depth);
    param.neigh.right = apply_pbc(m0, gi.pbc, row, col + 1, k, gi.rows, gi.cols, gi.depth);
    param.neigh.up = apply_pbc(m0, gi.pbc, row + 1, col, k, gi.rows, gi.cols, gi.depth);
    param.neigh.down = apply_pbc(m0, gi.pbc, row - 1, col, k, gi.rows, gi.cols, gi.depth);
    param.neigh.front = apply_pbc(m0, gi.pbc, row, col, k + 1, gi.rows, gi.cols, gi.depth);
    param.neigh.back = apply_pbc(m0, gi.pbc, row, col, k - 1, gi.rows, gi.cols, gi.depth);
    param.time = time;
#ifdef INCLUDE_DIPOLAR
    param.dipolar_energy = 0.0;
    for (int dr = -param.rows / 2; dr < param.rows / 2; ++dr) {
        double dy = dr * gi.lattice;
        for (int dc = -param.cols / 2; dc < param.cols / 2; ++dc) {
            double dx = dc * gi.lattice;
            for (int dk = -param.depth / 2; dk < param.depth / 2; ++dk) {
                if (dr == 0 && dc == 0 && dk == 0)
                    continue;
                double dz = dk * gi.lattice;
                double r_ij_ = sqrt(dx * dx + dy * dy + dz * dz);
                v3d r_ij = v3d_normalize(v3d_c(dx, dy, dz));

                v3d mj;
                grid_site_params gj;
                apply_pbc_complete(gs, m0, &mj, &gj, gi.pbc, row + dr, col + dc, k + dk, param.rows, param.cols, param.depth);

                double interaction = -MU_0 * param.gs.mu * gj.mu / (4.0 * M_PI);
                interaction *= (3.0 * v3d_dot(param.m, r_ij) - v3d_dot(param.m, mj)) / (r_ij_ * r_ij_ * r_ij_);
                param.dipolar_energy += interaction;
            }
        }
    }
#endif

    information_packed local_info = (information_packed){};

    local_info.exchange_energy = exchange_energy(param);
    local_info.dm_energy = dm_energy(param);
    local_info.field_energy = field_energy(param);
    local_info.anisotropy_energy = anisotropy_energy(param);
    local_info.cubic_energy = cubic_anisotropy_energy(param);
#ifdef INCLUDE_DIPOLAR
    local_info.dipolar_energy = param.dipolar_energy;
#endif
    local_info.energy = 0.5 * local_info.exchange_energy + 0.5 * local_info.dm_energy + local_info.field_energy + local_info.anisotropy_energy + local_info.cubic_energy + 0.5 * local_info.dipolar_energy;

    local_info.charge_finite = charge_derivative(param.m, param.neigh.left, param.neigh.right, param.neigh.up, param.neigh.down);
    local_info.charge_finite += charge_derivative(param.m, param.neigh.back, param.neigh.front, param.neigh.right, param.neigh.left);
    local_info.charge_finite += charge_derivative(param.m, param.neigh.down, param.neigh.up, param.neigh.front, param.neigh.back);

    local_info.charge_lattice = charge_lattice(param.m, param.neigh.left, param.neigh.right, param.neigh.up, param.neigh.down);
    local_info.charge_lattice += charge_lattice(param.m, param.neigh.back, param.neigh.front, param.neigh.right, param.neigh.left);
    local_info.charge_lattice += charge_lattice(param.m, param.neigh.down, param.neigh.up, param.neigh.front, param.neigh.back);

    local_info.abs_charge_finite = fabs(local_info.charge_finite);
    local_info.abs_charge_lattice = fabs(local_info.charge_lattice);
    local_info.avg_m = param.m;
    local_info.eletric_field = v3d_scalar(emergent_eletric_field(param.m, param.neigh.left, param.neigh.right, param.neigh.up, param.neigh.down, param.neigh.front, param.neigh.back, v3d_scalar(dm, 1.0 / dt), gi.lattice, gi.lattice, gi.lattice), gi.lattice * gi.lattice);
    local_info.magnetic_field_derivative = emergent_magnetic_field_derivative(param.m, param.neigh.left, param.neigh.right, param.neigh.up, param.neigh.down, param.neigh.front, param.neigh.back);
    local_info.magnetic_field_lattice = emergent_magnetic_field_lattice(param.m, param.neigh.left, param.neigh.right, param.neigh.up, param.neigh.down, param.neigh.front, param.neigh.back);
    local_info.charge_center_x = col * gi.lattice * local_info.charge_finite;
    local_info.charge_center_y = row * gi.lattice * local_info.charge_finite;
    local_info.charge_center_z = k * gi.lattice * local_info.charge_finite;
    local_info.abs_charge_center_x = col * gi.lattice * local_info.abs_charge_finite;
    local_info.abs_charge_center_y = row * gi.lattice * local_info.abs_charge_finite;
    local_info.abs_charge_center_z = k * gi.lattice * local_info.abs_charge_finite;
    v3d dm_dx = v3d_scalar(v3d_sub(param.neigh.right, param.neigh.left), 0.5);
    v3d dm_dy = v3d_scalar(v3d_sub(param.neigh.up, param.neigh.down), 0.5);
    v3d dm_dz = v3d_scalar(v3d_sub(param.neigh.front, param.neigh.back), 0.5);
    local_info.D_xx = v3d_dot(dm_dx, dm_dx);
    local_info.D_yy = v3d_dot(dm_dy, dm_dy);
    local_info.D_zz = v3d_dot(dm_dz, dm_dz);
    local_info.D_xy = v3d_dot(dm_dx, dm_dy);
    local_info.D_xz = v3d_dot(dm_dx, dm_dz);
    local_info.D_yz = v3d_dot(dm_dy, dm_dz);
    info[id] = local_info;
}

kernel void exchange_grid(GLOBAL v3d *to, GLOBAL v3d *from, unsigned int rows, unsigned int cols, unsigned int depth) {
    size_t id = get_global_id(0);
    if (id < (rows * cols * depth))
        to[id] = from[id];
}

kernel void render_grid_bwr(GLOBAL v3d *v, grid_info gi, unsigned int k,
                            GLOBAL RGBA32* rgba, unsigned int width, unsigned int height) {
    size_t id = get_global_id(0);

    if (id >= (width * height))
        return;

    int icol = id % width;
    int irow = id / width;
    int vcol = (float)icol / width * gi.cols;
    int vrow = (float)irow / height * gi.rows;

    //rendering inverts the grid, need to invert back
    vrow = gi.rows - 1 - vrow;

    if (vrow >= gi.rows || vcol >= gi.cols || k >= gi.depth)
        return;

    v3d m = v[k * gi.rows * gi.cols + vrow * gi.cols + vcol];

    rgba[id] = m_bwr_mapping(m);
}

kernel void render_grid_hsl(GLOBAL v3d *v, grid_info gi, unsigned int k,
                            GLOBAL RGBA32 *rgba, unsigned int width, unsigned int height) {
    size_t id = get_global_id(0);

    if (id >= (width * height))
        return;

    int icol = id % width;
    int irow = id / width;
    int vcol = (float)icol / width * gi.cols;
    int vrow = (float)irow / height * gi.rows;

    //rendering inverts the grid, need to invert back
    vrow = gi.rows - 1 - vrow;

    if (vrow >= gi.rows || vcol >= gi.cols || k >= gi.depth)
        return;

    v3d m = v[k * gi.rows * gi.cols + vrow * gi.cols + vcol];

    rgba[id] = m_to_hsl(m);
}

kernel void calculate_energy(GLOBAL grid_site_params *gs, GLOBAL v3d *v, grid_info gi, GLOBAL double *out, double time) {
    size_t id = get_global_id(0);

    if (id >= (gi.rows * gi.cols * gi.depth))
        return;

    int col = id % gi.cols;
    int row = id / gi.cols;
    int k = id / (gi.rows * gi.cols);

    parameters param;
    param.rows = gi.rows;
    param.cols = gi.cols;
    param.depth = gi.depth;
    param.gs = gs[id];
    param.m = v[id];
    param.lattice = gi.lattice;
    param.neigh.left = apply_pbc(v, gi.pbc, row, col - 1, k, gi.rows, gi.cols, gi.depth);
    param.neigh.right = apply_pbc(v, gi.pbc, row, col + 1, k, gi.rows, gi.cols, gi.depth);
    param.neigh.up = apply_pbc(v, gi.pbc, row + 1, col, k, gi.rows, gi.cols, gi.depth);
    param.neigh.down = apply_pbc(v, gi.pbc, row - 1, col, k, gi.rows, gi.cols, gi.depth);
    param.neigh.front = apply_pbc(v, gi.pbc, row, col, k + 1, gi.rows, gi.cols, gi.depth);
    param.neigh.back = apply_pbc(v, gi.pbc, row, col, k - 1, gi.rows, gi.cols, gi.depth);
    param.time = time;

#ifdef INCLUDE_DIPOLAR
    param.dipolar_energy = 0.0;
    for (int dr = -param.rows / 2; dr < param.rows / 2; ++dr) {
        double dy = dr * gi.lattice;
        for (int dc = -param.cols / 2; dc < param.cols / 2; ++dc) {
            double dx = dc * gi.lattice;
            for (int dk = -param.depth / 2; dk < param.depth / 2; ++dk) {
                if (dr == 0 && dc == 0 && dk == 0)
                    continue;
                double dz = dk * gi.lattice;
                double r_ij_ = sqrt(dx * dx + dy * dy + dz * dz);
                v3d r_ij = v3d_normalize(v3d_c(dx, dy, dz));

                v3d mj;
                grid_site_params gj;
                apply_pbc_complete(gs, m0, &mj, &gj, gi.pbc, row + dr, col + dc, k + dk, param.rows, param.cols, param.depth);

                double interaction = -MU_0 * param.gs.mu * gj.mu / (4.0 * M_PI);
                interaction *= (3.0 * v3d_dot(param.m, r_ij) - v3d_dot(param.m, mj)) / (r_ij_ * r_ij_ * r_ij_);
                param.dipolar_energy += interaction;
            }
        }
    }
#endif
    out[id] = energy(param);
}

kernel void thermal_step_gsa(GLOBAL grid_site_params *gs, GLOBAL v3d *v0, GLOBAL v3d *v1, grid_info gi, double qV, double gamma, double T, int seed) {
    size_t id = get_global_id(0);

    if (id >= (gi.rows * gi.cols * gi.depth))
        return;

    v3d v0l = v0[id];
    v3d v1l = v1[id];
    pinning pin = gs[id].pin;

    tyche_i_state state;
    tyche_i_seed(&state, seed + id);

    v3d delta = v3d_c(get_random_gsa(&state, qV, T, gamma),
                      get_random_gsa(&state, qV, T, gamma),
                      get_random_gsa(&state, qV, T, gamma));
    
    v1l = pin.pinned? pin.dir: v3d_normalize(v3d_sum(v0l, delta));

    v1[id] = v1l;
}

//0 -> before
//1 -> current
//2 -> new
kernel void gradient_descent_step(GLOBAL grid_site_params *gs, GLOBAL v3d *v0, GLOBAL v3d *v1, GLOBAL v3d *v2, grid_info gi,
                                  double mass, double T, double damping, double restoring, double dt, int seed) {
    size_t id = get_global_id(0);

    if (id >= (gi.rows * gi.cols * gi.depth))
        return;

    int col = id % gi.cols;
    int row = id / gi.cols;
    int k = id / (gi.rows * gi.cols);

    parameters param1 = (parameters){};
    param1.rows = gi.rows;
    param1.cols = gi.cols;
    param1.depth = gi.depth;
    param1.time = 0.0;
    param1.m = v1[id];
    param1.gs = gs[id];
    param1.lattice = gi.lattice;
    param1.neigh.left = apply_pbc(v1, gi.pbc, row, col - 1, k, gi.rows, gi.cols, gi.depth);
    param1.neigh.right = apply_pbc(v1, gi.pbc, row, col + 1, k, gi.rows, gi.cols, gi.depth);
    param1.neigh.up = apply_pbc(v1, gi.pbc, row + 1, col, k, gi.rows, gi.cols, gi.depth);
    param1.neigh.down = apply_pbc(v1, gi.pbc, row - 1, col, k, gi.rows, gi.cols, gi.depth);
    param1.neigh.front = apply_pbc(v1, gi.pbc, row, col, k + 1, gi.rows, gi.cols, gi.depth);
    param1.neigh.back = apply_pbc(v1, gi.pbc, row, col, k - 1, gi.rows, gi.cols, gi.depth);

#ifdef INCLUDE_DIPOLAR
    param1.dipolar_field = v3d_s(0.0);
    for (int dr = -param1.rows / 2; dr < param1.rows / 2; ++dr) {
        double dy = dr * gi.lattice;
        for (int dc = -param1.cols / 2; dc < param1.cols / 2; ++dc) {
            double dx = dc * gi.lattice;
            for (int dk = -param1.depth / 2; dk < param1.depth / 2; ++ dk) {
                if (dr == 0 && dc == 0 && dk == 0)
                    continue;
                double dz = dk * gi.lattice;
                double r_ij_ = sqrt(dx * dx + dy * dy + dz * dz);
                v3d r_ij = v3d_normalize(v3d_c(dx, dy, dz));
                v3d mj;
                grid_site_param1s gj;
                apply_pbc_complete(gs, v1, &mj, &gj, gi.pbc, row + dr, col + dc, k + dk, param1.rows, param1.cols, param1.depth);
                double factor = -MU_0 * param1.gs.mu * gj.mu / (4.0 * M_PI);
                v3d interaction = v3d_scalar(r_ij, 3.0 * v3d_dot(mj, r_ij));
                interaction = v3d_sub(interaction, mj);
                interaction = v3d_scalar(interaction, factor / (r_ij_ * r_ij_ * r_ij_));
                param1.dipolar_field = v3d_sum(param1.dipolar_field, interaction);
            }
        }
    }
#endif


    v3d v0l = v0[id];

    v3d dh_dm = effective_field(param1);
    v3d velocity = v3d_scalar(v3d_sub(param1.m, v0l), 1.0 / dt);
    v3d accel = v3d_scalar(param1.m, -restoring);
    accel = v3d_sum(accel, v3d_scalar(velocity, -damping));
    accel = v3d_sum(accel, dh_dm);
    if (!CLOSE_ENOUGH(T, 0.0, EPS)) {
        tyche_i_state state;
        tyche_i_seed(&state, seed + id);
        v3d temp = v3d_c(sqrt(T) * normal_distribution(&state),
                         sqrt(T) * normal_distribution(&state),
                         sqrt(T) * normal_distribution(&state));
        accel = v3d_sum(accel, temp);
    }
    accel = v3d_scalar(accel, 1.0 / mass);
    v2[id] = param1.gs.pin.pinned? param1.gs.pin.dir: v3d_normalize(v3d_sum(v3d_scalar(param1.m, 2.0), v3d_sub(v3d_scalar(accel, dt * dt), v0l)));
}
