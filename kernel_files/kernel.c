#include "constants.h"
#include "grid_types.h"
#include "simulation_funcs.h"

kernel void gpu_step(GLOBAL grid_site_params *gs, GLOBAL v3d *input, GLOBAL v3d *out, double dt, double time, grid_info gi) {
    size_t id = get_global_id(0);
    int col = id % gi.cols;
    int row = (id - col) / gi.cols;

    if (col >= gi.cols || row >= gi.rows) {
        return;
    }

    parameters param = (parameters){};
    param.rows = gi.rows;
    param.cols = gi.cols;
    param.gs = gs[id];
    param.m = apply_pbc(input, gi.pbc, row, col, gi.rows, gi.cols);
    param.neigh.left = apply_pbc(input, gi.pbc, row, col - 1, gi.rows, gi.cols);
    param.neigh.right = apply_pbc(input, gi.pbc, row, col + 1, gi.rows, gi.cols);
    param.neigh.up = apply_pbc(input, gi.pbc, row + 1, col, gi.rows, gi.cols);
    param.neigh.down = apply_pbc(input, gi.pbc, row - 1, col, gi.rows, gi.cols);
    param.time = time;
    tyche_i_state state;
    int seed = *((int*)(&time));
    seed = seed << 16;
    tyche_i_seed(&state, seed + id);
    param.state = &state;

#ifdef INCLUDE_DIPOLAR
    param.dipolar_field = v3d_s(0.0);
    for (int dr = -param.rows / 2; dr < param.rows / 2; ++dr) {
        double dy = dr * param.gs.lattice;
        for (int dc = -param.cols / 2; dc < param.cols / 2; ++dc) {
            if (dr == 0 && dc == 0)
                continue;
            double dx = dc * param.gs.lattice;
            double r_ij_ = sqrt(dx * dx + dy * dy);
            v3d r_ij = v3d_normalize(v3d_c(dx, dy, 0));
            v3d mj;
            grid_site_params gj;
            apply_pbc_complete(gs, input, &mj, &gj, gi.pbc, row + dr, col + dc, param.rows, param.cols);
            double factor = -MU_0 * param.gs.mu * gj.mu / (4.0 * M_PI);
            v3d interaction = v3d_scalar(r_ij, 3.0 * v3d_dot(mj, r_ij));
            interaction = v3d_sub(interaction, mj);
            interaction = v3d_scalar(interaction, factor / (r_ij_ * r_ij_ * r_ij_));
            param.dipolar_field = v3d_sum(param.dipolar_field, interaction);
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
    int col = id % gi.cols;
    int row = (id - col) / gi.cols;

    if (col >= gi.cols || row >= gi.rows)
        return;

    parameters param;
    param.rows = gi.rows;
    param.cols = gi.cols;
    param.gs = gs[id];
    param.m = m0[id];
    v3d dm = v3d_sub(m1[id], param.m);
    param.neigh.left = apply_pbc(m0, gi.pbc, row, col - 1, gi.rows, gi.cols);
    param.neigh.right = apply_pbc(m0, gi.pbc, row, col + 1, gi.rows, gi.cols);
    param.neigh.up = apply_pbc(m0, gi.pbc, row + 1, col, gi.rows, gi.cols);
    param.neigh.down = apply_pbc(m0, gi.pbc, row - 1, col, gi.rows, gi.cols);
    param.time = time;
#ifdef INCLUDE_DIPOLAR
    param.dipolar_energy = 0.0;
    for (int dr = -param.rows / 2; dr < param.rows / 2; ++dr) {
        double dy = dr * param.gs.lattice;
        for (int dc = -param.cols / 2; dc < param.cols / 2; ++dc) {
            if (dr == 0 && dc == 0)
                continue;
            double dx = dc * param.gs.lattice;
            double r_ij_ = sqrt(dx * dx + dy * dy);
            v3d r_ij = v3d_normalize(v3d_c(dx, dy, 0));

            v3d mj;
            grid_site_params gj;
            apply_pbc_complete(gs, m0, &mj, &gj, gi.pbc, row + dr, col + dc, param.rows, param.cols);

            double interaction = -MU_0 * param.gs.mu * gj.mu / (4.0 * M_PI);
            interaction *= (3.0 * v3d_dot(param.m, r_ij) - v3d_dot(param.m, mj)) / (r_ij_ * r_ij_ * r_ij_);
            param.dipolar_energy += interaction;
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
    local_info.charge_lattice = charge_lattice(param.m, param.neigh.left, param.neigh.right, param.neigh.up, param.neigh.down);
    local_info.avg_m = param.m;
    local_info.eletric_field = v3d_scalar(emergent_eletric_field(param.m, param.neigh.left, param.neigh.right, param.neigh.up, param.neigh.down, v3d_scalar(dm, 1.0 / dt), param.gs.lattice, param.gs.lattice), param.gs.lattice * dt);
    local_info.magnetic_field_derivative = emergent_magnetic_field_derivative(param.m, param.neigh.left, param.neigh.right, param.neigh.up, param.neigh.down);
    local_info.magnetic_field_lattice = emergent_magnetic_field_lattice(param.m, param.neigh.left, param.neigh.right, param.neigh.up, param.neigh.down);
    local_info.charge_center_x = col * param.gs.lattice * local_info.charge_finite;
    local_info.charge_center_y = row * param.gs.lattice * local_info.charge_finite;
    info[id] = local_info;
}

kernel void exchange_grid(GLOBAL v3d *to, GLOBAL v3d *from) {
    size_t id = get_global_id(0);
    to[id] = from[id];
}

kernel void render_grid_bwr(GLOBAL v3d *v, grid_info gi,
                            GLOBAL RGBA32* rgba, unsigned int width, unsigned int height) {
    size_t id = get_global_id(0);
    int icol = id % width;
    int irow = (id - icol) / width;
    int vcol = (float)icol / width * gi.cols;
    int vrow = (float)irow / height * gi.rows;

    //rendering inverts the grid, need to invert back
    vrow = gi.rows - 1 - vrow;

    if (vrow >= gi.rows || vcol >= gi.cols || icol >= width || irow >= height)
        return;

    v3d m = v[vrow * gi.cols + vcol];

    rgba[id] = m_bwr_mapping(m);
}

kernel void render_grid_hsl(GLOBAL v3d *v, grid_info gi,
                            GLOBAL RGBA32 *rgba, unsigned int width, unsigned int height) {
    size_t id = get_global_id(0);
    int icol = id % width;
    int irow = (id - icol) / width;
    int vcol = (float)icol / width * gi.cols;
    int vrow = (float)irow / height * gi.rows;

    //rendering inverts the grid, need to invert back
    vrow = gi.rows - 1 - vrow;

    if (vrow >= gi.rows || vcol >= gi.cols || icol >= width || irow >= height)
        return;

    v3d m = v[vrow * gi.cols + vcol];

    rgba[id] = m_to_hsl(m);
}

kernel void calculate_charge_to_render(GLOBAL v3d *v, grid_info gi, GLOBAL double *out) {
    size_t id = get_global_id(0);
    int col = id % gi.cols;
    int row = (id - col) / gi.cols;

    if (col >= gi.cols || row >= gi.rows)
        return;

    v3d m = v[id];
    v3d left = apply_pbc(v, gi.pbc, row, col - 1, gi.rows, gi.cols);
    v3d right = apply_pbc(v, gi.pbc, row, col + 1, gi.rows, gi.cols);
    v3d up = apply_pbc(v, gi.pbc, row + 1, col, gi.rows, gi.cols);
    v3d down = apply_pbc(v, gi.pbc, row - 1, col, gi.rows, gi.cols);

    out[id] = charge_lattice(m, left, right, up, down);
}

kernel void render_charge(GLOBAL double *input, unsigned int rows, unsigned int cols, double charge_min, double charge_max,
                          GLOBAL RGBA32 *rgba, unsigned int width, unsigned int height) {
    size_t id = get_global_id(0);
    int icol = id % width;
    int irow = (id - icol) / width;
    int vcol = (float)icol / width * cols;
    int vrow = (float)irow / height * rows;

    //rendering inverts the grid, need to invert back
    vrow = rows - 1 - vrow;

    if (vrow >= rows || vcol >= cols || icol >= width || irow >= height)
        return;

    v3d start = v3d_c(0, 0, 0);
    v3d middle = v3d_c(0.5, 0.5, 0.5);
    v3d end = v3d_c(1, 1, 1);

    double charge = input[vrow * cols + vcol];
    charge = (charge - charge_min) / (charge_max - charge_min);
    rgba[id] = linear_mapping(clamp(charge, 0.0, 1.0), start, middle, end);
}

kernel void render_pinning(GLOBAL grid_site_params *input, unsigned int rows, unsigned int cols,
                          GLOBAL RGBA32 *rgba, unsigned int width, unsigned int height) {
    size_t id = get_global_id(0);
    int icol = id % width;
    int irow = (id - icol) / width;
    int vcol = (float)icol / width * cols;
    int vrow = (float)irow / height * rows;

    //rendering inverts the grid, need to invert back
    vrow = rows - 1 - vrow;

    if (vrow >= rows || vcol >= cols || icol >= width || irow >= height)
        return;
    if (input[vrow * cols + vcol].pin.pinned)
        rgba[id] = (RGBA32){.a = 0xff, .b = 0, .g = 0xff, .r = 0xff};
}

kernel void calculate_energy(GLOBAL grid_site_params *gs, GLOBAL v3d *v, grid_info gi, GLOBAL double *out, double time) {
    size_t id = get_global_id(0);
    int col = id % gi.cols;
    int row = (id - col) / gi.cols;

    if (col >= gi.cols || row >= gi.rows)
        return;

    parameters param;
    param.rows = gi.rows;
    param.cols = gi.cols;
    param.gs = gs[id];
    param.m = v[id];
    param.neigh.left = apply_pbc(v, gi.pbc, row, col - 1, gi.rows, gi.cols);
    param.neigh.right = apply_pbc(v, gi.pbc, row, col + 1, gi.rows, gi.cols);
    param.neigh.up = apply_pbc(v, gi.pbc, row + 1, col, gi.rows, gi.cols);
    param.neigh.down = apply_pbc(v, gi.pbc, row - 1, col, gi.rows, gi.cols);
    param.time = time;

#ifdef INCLUDE_DIPOLAR
    param.dipolar_energy = 0.0;
    for (int dr = -param.rows / 2; dr < param.rows / 2; ++dr) {
        double dy = dr * param.gs.lattice;
        for (int dc = -param.cols / 2; dc < param.cols / 2; ++dc) {
            if (dr == 0 && dc == 0)
                continue;
            double dx = dc * param.gs.lattice;
            double r_ij_ = sqrt(dx * dx + dy * dy);
            v3d r_ij = v3d_normalize(v3d_c(dx, dy, 0));

            v3d mj;
            grid_site_params gj;
            apply_pbc_complete(gs, v, &mj, &gj, gi.pbc, row + dr, col + dc, param.rows, param.cols);

            double interaction = -MU_0 * param.gs.mu * gj.mu / (4.0 * M_PI);
            interaction *= (3.0 * v3d_dot(param.m, r_ij) - v3d_dot(param.m, mj)) / (r_ij_ * r_ij_ * r_ij_);
            param.dipolar_energy += interaction;
        }
    }
#endif

    out[id] = energy(param);
}

kernel void render_energy(GLOBAL double *ene, unsigned int rows, unsigned int cols, double energy_min, double energy_max,
                                 GLOBAL RGBA32 *rgba, unsigned int width, unsigned int height) {
    size_t id = get_global_id(0);
    int icol = id % width;
    int irow = (id - icol) / width;
    int vcol = (float)icol / width * cols;
    int vrow = (float)irow / height * rows;

    //rendering inverts the grid, need to invert back
    vrow = rows - 1 - vrow;

    if (vrow >= rows || vcol >= cols || icol >= width || irow >= height)
        return;

    v3d start = v3d_c(0, 0, 0);
    v3d middle = v3d_c(0.5, 0.5, 0.5);
    v3d end = v3d_c(1, 1, 1);

    double energy = ene[vrow * cols + vcol];
    energy = (energy - energy_min) / (energy_max - energy_min);
    rgba[id] = linear_mapping(clamp(energy, 0.0, 1.0), start, middle, end);
}

kernel void thermal_step_gsa(GLOBAL grid_site_params *gs, GLOBAL v3d *v0, GLOBAL v3d *v1, grid_info gi, double qV, double gamma, double T, int seed) {
    size_t i = get_global_id(0);
    v3d v0l = v0[i];
    v3d v1l = v1[i];
    pinning pin = gs[i].pin;

    tyche_i_state state;
    tyche_i_seed(&state, seed + i);

    v3d delta = v3d_c(get_random_gsa(&state, qV, T, gamma),
                      get_random_gsa(&state, qV, T, gamma),
                      get_random_gsa(&state, qV, T, gamma));
    
    v1l = pin.pinned? pin.dir: v3d_normalize(v3d_sum(v0l, delta));

    v1[i] = v1l;
}

//0 -> before
//1 -> current
//2 -> new
kernel void gradient_descent_step(GLOBAL grid_site_params *gs, GLOBAL v3d *v0, GLOBAL v3d *v1, GLOBAL v3d *v2, grid_info gi,
                                  double mass, double T, double damping, double restoring, double dt, int seed) {
    size_t id = get_global_id(0);
    int col = id % gi.cols;
    int row = (id - col) / gi.cols;
    if (col >= gi.cols || row >= gi.rows)
        return;

    parameters param1 = (parameters){};
    param1.rows = gi.rows;
    param1.cols = gi.cols;
    param1.time = 0.0;
    param1.m = v1[id];
    param1.gs = gs[id];
    param1.neigh.up = apply_pbc(v1, gi.pbc, row + 1, col, gi.rows, gi.cols);
    param1.neigh.down = apply_pbc(v1, gi.pbc, row - 1, col, gi.rows, gi.cols);
    param1.neigh.right = apply_pbc(v1, gi.pbc, row, col + 1, gi.rows, gi.cols);
    param1.neigh.left = apply_pbc(v1, gi.pbc, row, col - 1, gi.rows, gi.cols);

#ifdef INCLUDE_DIPOLAR
    param1.dipolar_field = v3d_s(0.0);
    for (int dr = -param1.rows / 2; dr < param1.rows / 2; ++dr) {
        double dy = dr * param1.gs.lattice;
        for (int dc = -param1.cols / 2; dc < param1.cols / 2; ++dc) {
            if (dr == 0 && dc == 0)
                continue;
            double dx = dc * param1.gs.lattice;
            double r_ij_ = sqrt(dx * dx + dy * dy);
            v3d r_ij = v3d_normalize(v3d_c(dx, dy, 0));
            v3d mj;
            grid_site_params gj;
            apply_pbc_complete(gs, v1, &mj, &gj, gi.pbc, row + dr, col + dc, param1.rows, param1.cols);
            double factor = -MU_0 * param1.gs.mu * gj.mu / (4.0 * M_PI);
            v3d interaction = v3d_scalar(r_ij, 3.0 * v3d_dot(mj, r_ij));
            interaction = v3d_sub(interaction, mj);
            interaction = v3d_scalar(interaction, factor / (r_ij_ * r_ij_ * r_ij_));
            param1.dipolar_field = v3d_sum(param1.dipolar_field, interaction);
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
