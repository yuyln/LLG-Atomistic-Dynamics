#include "constants.h"
#include "grid_types.h"
#include "simulation_funcs.h"

double nsrandom(tyche_i_state *state, double start, double end) {
    return tyche_i_double((*state)) * (end - start) + start;
}

char4 linear_mapping(double t, double3 start, double3 middle, double3 end) {
    double3 color;
    if (t < 0.5) {
        color = (middle - start) * 2.0 * t + start;
    } else {
        color = (end - middle) * (2.0 * t - 1.0) + middle;
    }
    //RGBA -> BGRA
    return (char4){color.z * 255, color.y * 255, color.x * 255, 255};
}

char4 m_bwr_mapping(v3d m) {
    double3 start = {0x03 / 255.0, 0x7f / 255.0, 0xff / 255.0};
    double3 middle = {1, 1, 1};
    double3 end = {0xf4 / 255.0, 0x05 / 255.0, 0x01 / 255.0};

    //m = v3d_normalize(m);
    double mz = m.z;

    return linear_mapping(0.5 * mz + 0.5, start, middle, end);
}

double _v(double m1, double m2, double hue) {
    //hue = hue % 1.0;
    int hue_i = floor(hue);
    hue = hue - hue_i;
    if (hue < (1.0 / 6.0))
        return m1 + (m2 - m1) * hue * 6.0;
    if (hue < 0.5)
        return m2;
    if (hue < (2.0 / 3.0))
        return m1 + (m2 - m1) * (2.0 / 3.0 - hue) * 6.0;
    return m1;
}

char4 hsl_to_rgb(double h, double s, double l) {
    if (CLOSE_ENOUGH(s, 0.0, EPS))
        return (char4){255 * l, 255 * l, 255 * l, 255};
    double m2;
    if (l <= 0.5)
        m2 = l * (1.0 + s);
    else
        m2 = l + s - l * s;
    double m1 = 2.0 * l - m2;
    char4 ret = (char4){_v(m1, m2, h + 1.0 / 3.0) * 255, _v(m1, m2, h) * 255, _v(m1, m2, h - 1.0 / 3.0) * 255, 255};
    //RGBA -> BGRA
    return (char4){ret.z, ret.y, ret.x, ret.w};
}

char4 m_to_hsl(v3d m) {
    //m = v3d_normalize(m);
    double angle = atan2(m.y, m.x) / M_PI;
    angle = (angle + 1.0) / 2.0;
    double l = (m.z + 1.0) / 2.0;
    double s = 1.0;
    return hsl_to_rgb(angle, s, l);
}

//Assume D=1
double get_random_gsa(tyche_i_state *state, double qV, double T, double gamma) {
    double dx = nsrandom(state, -10, 10);
    double c = sqrt((qV - 1.0) / M_PI) * gamma * pow(T, -1.0 / (3.0 - qV));
    double l = pow(1.0 + (qV - 1) * dx * dx / pow(T, 2.0 / (3.0 - qV)), 1.0 / (qV - 1.0));
    return c * dx / l;
}


kernel void gpu_step(GLOBAL grid_site_param *gs, GLOBAL v3d *input, GLOBAL v3d *out, double dt, double time, grid_info gi) {
    size_t id = get_global_id(0);
    int col = id % gi.cols;
    int row = (id - col) / gi.cols;

    if (col >= gi.cols || row >= gi.rows) {
        return;
    }

    parameters param;
    param.gs = gs[id];
    param.m = apply_pbc(input, gi.pbc, row, col, gi.rows, gi.cols);
    param.neigh.left = apply_pbc(input, gi.pbc, row, col - 1, gi.rows, gi.cols);
    param.neigh.right = apply_pbc(input, gi.pbc, row, col + 1, gi.rows, gi.cols);
    param.neigh.up = apply_pbc(input, gi.pbc, row + 1, col, gi.rows, gi.cols);
    param.neigh.down = apply_pbc(input, gi.pbc, row - 1, col, gi.rows, gi.cols);
    param.time = time;

    out[id] = v3d_normalize(param.gs.pin.pinned? param.gs.pin.dir: v3d_sum(param.m, step(param, dt)));
}

kernel void extract_info(GLOBAL grid_site_param *gs, GLOBAL v3d *m0, GLOBAL v3d *m1, GLOBAL information_packed *info, double dt, double time, grid_info gi) {
    size_t id = get_global_id(0);
    int col = id % gi.cols;
    int row = (id - col) / gi.cols;

    if (col >= gi.cols || row >= gi.rows)
        return;

    parameters param;
    param.gs = gs[id];
    param.m = m0[id];
    v3d dm = v3d_sub(m1[id], param.m);
    param.neigh.left = apply_pbc(m0, gi.pbc, row, col - 1, gi.rows, gi.cols);
    param.neigh.right = apply_pbc(m0, gi.pbc, row, col + 1, gi.rows, gi.cols);
    param.neigh.up = apply_pbc(m0, gi.pbc, row + 1, col, gi.rows, gi.cols);
    param.neigh.down = apply_pbc(m0, gi.pbc, row - 1, col, gi.rows, gi.cols);
    param.time = time;
    information_packed local_info = {0};

    local_info.exchange_energy = exchange_energy(param);
    local_info.dm_energy = dm_energy(param);
    local_info.field_energy = field_energy(param);
    local_info.anisotropy_energy = anisotropy_energy(param);
    local_info.cubic_energy = cubic_anisotropy_energy(param);
    local_info.energy = 0.5 * local_info.exchange_energy + 0.5 * local_info.dm_energy + local_info.field_energy + local_info.anisotropy_energy + local_info.cubic_energy;
    local_info.charge_finite = charge_derivative(param.m, param.neigh.left, param.neigh.right, param.neigh.up, param.neigh.down);
    local_info.charge_lattice = charge_lattice(param.m, param.neigh.left, param.neigh.right, param.neigh.up, param.neigh.down);
    local_info.avg_m = param.m;
    local_info.eletric_field = v3d_scalar(emergent_eletric_field(param.m, param.neigh.left, param.neigh.right, param.neigh.up, param.neigh.down, v3d_scalar(dm, 1.0 / dt), param.gs.lattice, param.gs.lattice), param.gs.lattice * dt);
    local_info.magnetic_field_derivative = emergent_magnetic_field_derivative(param.m, param.neigh.left, param.neigh.right, param.neigh.up, param.neigh.down);
    local_info.magnetic_field_lattice = emergent_magnetic_field_lattice(param.m, param.neigh.left, param.neigh.right, param.neigh.up, param.neigh.down);
    info[id] = local_info;
}

kernel void exchange_grid(GLOBAL v3d *to, GLOBAL v3d *from) {
    size_t id = get_global_id(0);
    to[id] = from[id];
}

kernel void render_grid_bwr(GLOBAL v3d *v, grid_info gi,
                        GLOBAL char4 *rgba, unsigned int width, unsigned int height) {
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
                        GLOBAL char4 *rgba, unsigned int width, unsigned int height) {
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
                                      GLOBAL char4 *rgba, unsigned int width, unsigned int height) {
    size_t id = get_global_id(0);
    int icol = id % width;
    int irow = (id - icol) / width;
    int vcol = (float)icol / width * cols;
    int vrow = (float)irow / height * rows;

    //rendering inverts the grid, need to invert back
    vrow = rows - 1 - vrow;

    if (vrow >= rows || vcol >= cols || icol >= width || irow >= height)
        return;

    double3 start = {0, 0, 0};
    double3 middle = {0.5, 0.5, 0.5};
    double3 end = {1, 1, 1};

    double charge = input[vrow * cols + vcol];
    charge = (charge - charge_min) / (charge_max - charge_min);
    rgba[id] = linear_mapping(clamp(charge, 0.0, 1.0), start, middle, end);
}

kernel void calculate_energy(GLOBAL grid_site_param *gs, GLOBAL v3d *v, grid_info gi, GLOBAL double *out, double time) {
    size_t id = get_global_id(0);
    int col = id % gi.cols;
    int row = (id - col) / gi.cols;

    if (col >= gi.cols || row >= gi.rows)
        return;

    parameters param;
    param.gs = gs[id];
    param.m = v[id];
    param.neigh.left = apply_pbc(v, gi.pbc, row, col - 1, gi.rows, gi.cols);
    param.neigh.right = apply_pbc(v, gi.pbc, row, col + 1, gi.rows, gi.cols);
    param.neigh.up = apply_pbc(v, gi.pbc, row + 1, col, gi.rows, gi.cols);
    param.neigh.down = apply_pbc(v, gi.pbc, row - 1, col, gi.rows, gi.cols);
    param.time = time;

    out[id] = energy(param);
}

kernel void render_energy(GLOBAL double *ene, unsigned int rows, unsigned int cols, double energy_min, double energy_max,
                                 GLOBAL char4 *rgba, unsigned int width, unsigned int height) {
    size_t id = get_global_id(0);
    int icol = id % width;
    int irow = (id - icol) / width;
    int vcol = (float)icol / width * cols;
    int vrow = (float)irow / height * rows;

    //rendering inverts the grid, need to invert back
    vrow = rows - 1 - vrow;

    if (vrow >= rows || vcol >= cols || icol >= width || irow >= height)
        return;

    double3 start = {0, 0, 0};
    double3 middle = {0.5, 0.5, 0.5};
    double3 end = {1, 1, 1};

    double energy = ene[vrow * cols + vcol];
    energy = (energy - energy_min) / (energy_max - energy_min);
    rgba[id] = linear_mapping(clamp(energy, 0.0, 1.0), start, middle, end);
}

kernel void thermal_step_gsa(GLOBAL grid_site_param *gs, GLOBAL v3d *v0, GLOBAL v3d *v1, grid_info gi, double qV, double gamma, double T, int seed) {
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
