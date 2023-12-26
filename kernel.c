#include "grid_types.h"
#include "simulation_funcs.h"

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

kernel void exchange_grid(GLOBAL v3d *to, GLOBAL v3d *from) {
    size_t id = get_global_id(0);
    to[id] = from[id];
}

kernel void v3d_to_rgb(GLOBAL v3d *input, GLOBAL char4 *rgb) {
    double3 start = {0x03 / 255.0, 0x7f / 255.0, 0xff / 255.0};
    double3 middle = {1, 1, 1};
    double3 end = {0xf4 / 255.0, 0x05 / 255.0, 0x01 / 255.0};
    size_t id = get_global_id(0);
    v3d m = input[id];
    m = v3d_normalize(m);
    double mz = m.z;
    double mapping = (mz + 1.0) / 2.0;
    double3 color;
    if (mapping < 0.5) {
        color = (middle - start) * 2.0 * mapping + start;
    } else {
        color = (end - middle) * (2.0 * mapping - 1.0) + middle;
    }
    char4 ret = {color.x * 255, color.y * 255, color.z * 255, 255};
    rgb[id] = ret;
}

char4 mz_linear_mapping(double mz, double3 start, double3 middle, double3 end) {
    double mapping = (mz + 1.0) / 2.0;
    double3 color;
    if (mapping < 0.5) {
        color = (middle - start) * 2.0 * mapping + start;
    } else {
        color = (end - middle) * (2.0 * mapping - 1.0) + middle;
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

    return mz_linear_mapping(mz, start, middle, end);
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

kernel void render_grid(GLOBAL v3d *input, unsigned int rows, unsigned int cols,
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

    v3d m = input[vrow * cols + vcol];

    rgba[id] = m_to_hsl(m);
}
