#include "grid_types.h"
#include "simulation_funcs.h"

kernel void gpu_step(GLOBAL grid_site_param *gs, GLOBAL v3d *input, GLOBAL v3d *out, double dt, double time, pbc_rules pbc) {
    size_t id = get_global_id(0);
    grid_site_param gsp = gs[id];
    int col = id % gsp.sz.cols;
    int row = ((id - col) / gsp.sz.cols) % gsp.sz.rows;
    int depth = (id - col - row * gsp.sz.cols) / (gsp.sz.cols * gsp.sz.rows);
    parameters param;
    param.gs = gsp;
    param.c = apply_pbc(input, gsp.sz, (matrix_loc){.row = row, .col = col, .depth = depth}, pbc);
    param.l = apply_pbc(input, gsp.sz, (matrix_loc){.row = row, .col = col - 1, .depth = depth}, pbc);
    param.l = apply_pbc(input, gsp.sz, (matrix_loc){.row = row, .col = col + 1, .depth = depth}, pbc);
    param.u = apply_pbc(input, gsp.sz, (matrix_loc){.row = row + 1, .col = col, .depth = depth}, pbc);
    param.d = apply_pbc(input, gsp.sz, (matrix_loc){.row = row - 1, .col = col, .depth = depth}, pbc);
#ifndef NBULK
    param.f = apply_pbc(input, gsp.sz, (matrix_loc){.row = row, .col = col, .depth = depth + 1}, pbc);
    param.b = apply_pbc(input, gsp.sz, (matrix_loc){.row = row, .col = col, .depth = depth - 1}, pbc);
#endif
    param.time = time;

    out[id] = v3d_normalize(v3d_sum(param.c, dm_dt(param)));
}

kernel void exchange_grid(GLOBAL v3d *to, GLOBAL v3d *from) {
    size_t id = get_global_id(0);
    to[id] = from[id];
}
