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

    out[id] = v3d_normalize(v3d_sum(param.m, step(param, dt)));
}

kernel void exchange_grid(GLOBAL v3d *to, GLOBAL v3d *from) {
    size_t id = get_global_id(0);
    to[id] = from[id];
}

kernel void v3d_to_rgb(GLOBAL v3d *input, GLOBAL char4 *rgb) {
    double3 start = {0, 0, 1};
    double3 middle = {1, 1, 1};
    double3 end = {1, 0, 0};
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
