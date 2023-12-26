#include "constants.h"
#include "grid_types.h"
#include "simulation_funcs.h"

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

    m = v3d_normalize(m);
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
    m = v3d_normalize(m);
    double angle = atan2(m.y, m.x) / M_PI;
    angle = (angle + 1.0) / 2.0;
    double l = (m.z + 1.0) / 2.0;
    double s = 1.0;
    return hsl_to_rgb(angle, s, l);
}


kernel void gpu_step(GLOBAL grid_site_param *gs, GLOBAL v3d *input, GLOBAL v3d *out, double dt, double time, grid_info gi) {
    size_t id = get_global_id(0);
    int col = id % gi.cols;
    int row = (id - col) / gi.cols;

    if (col >= gi.cols || row >= gi.rows)
        return;

    parameters param;
    param.gs = gs[id];
    param.m = input[id];
    param.neigh.left = apply_pbc(input, gi, row, col - 1);
    param.neigh.right = apply_pbc(input, gi, row, col + 1);
    param.neigh.up = apply_pbc(input, gi, row + 1, col);
    param.neigh.down = apply_pbc(input, gi, row - 1, col);
    param.time = time;

    out[id] =  v3d_normalize(param.gs.pin.pinned? param.gs.pin.dir: v3d_sum(param.m, step(param, dt)));
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
    param.neigh.left = apply_pbc(m0, gi, row, col - 1);
    param.neigh.right = apply_pbc(m0, gi, row, col + 1);
    param.neigh.up = apply_pbc(m0, gi, row + 1, col);
    param.neigh.down = apply_pbc(m0, gi, row - 1, col);
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

kernel void v3d_to_rgb(GLOBAL v3d *input, GLOBAL char4 *rgb) {
    size_t id = get_global_id(0);
    v3d m = input[id];
    //char4 ret = m_bwr_mapping(m);
    char4 ret = m_to_hsl(m);
    //BGRA -> RGBA 
    rgb[id] = (char4){ret.z, ret.y, ret.x, ret.w};
}


kernel void render_grid(GLOBAL v3d *input, grid_info gi, 
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

    v3d m = input[vrow * gi.cols + vcol];

    rgba[id] = m_to_hsl(m);
}

kernel void render_grid_compa(GLOBAL information_packed *info, grid_info gi, double charge_min, double charge_max,
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

    v3d m = info[vrow * gi.cols + vcol].avg_m;

    rgba[id] = m_to_hsl(m);
}

kernel void render_topological_charge(GLOBAL information_packed *info, grid_info gi, double charge_min, double charge_max,
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

    double3 start = {0, 0, 0};
    double3 middle = {0.5, 0.5, 0.5};
    double3 end = {1, 1, 1};

    double charge = info[vrow * gi.cols + vcol].charge_lattice;
    charge = (charge - charge_min) / (charge_max - charge_min);
    rgba[id] = linear_mapping(clamp(charge, 0.0, 1.0), start, middle, end);
}

kernel void render_magnetic_field(GLOBAL information_packed *info, grid_info gi, double field_min, double field_max,
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

    v3d field = info[vrow * gi.cols + vcol].magnetic_field_lattice;
    field.x = (field.x - field_min) / (field_max - field_min);
    field.y = (field.y - field_min) / (field_max - field_min);
    field.z = (field.z - field_min) / (field_max - field_min);
    rgba[id] = m_to_hsl(field);
}

kernel void render_eletric_field(GLOBAL information_packed *info, grid_info gi, double field_min, double field_max,
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

    v3d field = info[vrow * gi.cols + vcol].eletric_field;
    //field.x = (field.x - field_min) / (field_max - field_min);
    //field.y = (field.y - field_min) / (field_max - field_min);
    //field.z = (field.z - field_min) / (field_max - field_min);
    rgba[id] = m_to_hsl(v3d_normalize(field));
}

kernel void render_energy(GLOBAL information_packed *info, grid_info gi, double energy_min, double energy_max,
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

    double3 start = {0, 0, 0};
    double3 middle = {0.5, 0.5, 0.5};
    double3 end = {1, 1, 1};

    double energy = info[vrow * gi.cols + vcol].energy;
    energy = (energy - energy_min) / (energy_max - energy_min);
    rgba[id] = linear_mapping(clamp(energy, 0.0, 1.0), start, middle, end);
}
