#include "simulation_funcs.h"

v3d apply_pbc(GLOBAL v3d *v, grid_info info, int row, int col) {
    int pbc_x = info.pbc.dirs & (1 << 0);
    int pbc_y = info.pbc.dirs & (1 << 1);

    if (row >= (int)info.rows || row < 0) {
        if (!pbc_y)
            return info.pbc.m;
        row = ((row % info.rows) + info.rows) % info.rows;
    }

    if (col >= (int)info.cols || col < 0) {
        if (!pbc_x)
            return info.pbc.m;
        col = ((col % info.cols) + info.cols) % info.cols;
    }

    return v[row * info.cols + col];
}

v3d get_dm_vec(v3d dr, double dm, dm_symmetry dm_sym) {
    switch (dm_sym) {
        case R_ij_CROSS_Z: {
            return v3d_scalar(v3d_cross(dr, v3d_c(0, 0, 1)), dm);
        }
        case R_ij: {
            return v3d_scalar(dr, dm);
        }
    }
    return v3d_s(0);
}

v3d generate_magnetic_field(grid_site_param gs, double time) {
    UNUSED(gs);
    UNUSED(time);
    double normalized = 0.5 * gs.dm * gs.dm / gs.exchange;
    double real = normalized / gs.mu;
    return v3d_c(0, 0, real);
}

current generate_current(grid_site_param gs, double time) {
    UNUSED(gs);
    UNUSED(time);
    return (current) {0};
}

double exchange_energy(parameters param) {
    return -(v3d_dot(param.m, param.neigh.left) + v3d_dot(param.m, param.neigh.right) +
             v3d_dot(param.m, param.neigh.up) + v3d_dot(param.m, param.neigh.down)) * param.gs.exchange;
}

double dm_energy(parameters param) {
    double ret = 0.0;
    ret += v3d_dot(get_dm_vec(v3d_c(1, 0, 0), param.gs.dm + param.gs.dm_ani, param.gs.dm_sym), v3d_cross(param.m, param.neigh.right));
    ret += v3d_dot(get_dm_vec(v3d_c(-1, 0, 0), param.gs.dm + param.gs.dm_ani, param.gs.dm_sym), v3d_cross(param.m, param.neigh.left));
    ret += v3d_dot(get_dm_vec(v3d_c(0, 1, 0), param.gs.dm - param.gs.dm_ani, param.gs.dm_sym), v3d_cross(param.m, param.neigh.up));
    ret += v3d_dot(get_dm_vec(v3d_c(0, -1, 0), param.gs.dm - param.gs.dm_ani, param.gs.dm_sym), v3d_cross(param.m, param.neigh.down));
    return -ret;
}

double anisotropy_energy(parameters param) {
    double scalar = v3d_dot(param.m, param.gs.ani.dir);
    return -param.gs.ani.ani * scalar * scalar;
}

//double cubic_anisotropy_energy(parameters param);

double field_energy(parameters param) {
    return -param.gs.mu * v3d_dot(generate_magnetic_field(param.gs, param.time), param.m);
}

double energy(parameters param) {
    return 0.5 * exchange_energy(param) + 0.5 * dm_energy(param) + anisotropy_energy(param) + field_energy(param) /*+ cubic_anisotropy_energy(param)*/;
}

v3d effective_field(parameters param) {
    v3d ret = v3d_s(0);

    v3d exchange_field = v3d_sum(v3d_sum(param.neigh.right, param.neigh.left), v3d_sum(param.neigh.up, param.neigh.down));
    exchange_field = v3d_scalar(exchange_field, param.gs.exchange);
    ret = v3d_sub(ret, exchange_field);


    v3d dm_field_x = v3d_sum(v3d_cross(get_dm_vec(v3d_c(1, 0, 0), param.gs.dm + param.gs.dm_ani, param.gs.dm_sym), param.neigh.right),
                             v3d_cross(get_dm_vec(v3d_c(-1, 0, 0), param.gs.dm + param.gs.dm_ani, param.gs.dm_sym), param.neigh.left));

    v3d dm_field_y = v3d_sum(v3d_cross(get_dm_vec(v3d_c(0, 1, 0), param.gs.dm - param.gs.dm_ani, param.gs.dm_sym), param.neigh.up),
                             v3d_cross(get_dm_vec(v3d_c(0, -1, 0), param.gs.dm - param.gs.dm_ani, param.gs.dm_sym), param.neigh.down));
    v3d dm_field = v3d_sum(dm_field_x, dm_field_y);

    ret = v3d_sub(ret, dm_field);

    ret = v3d_sub(ret, v3d_scalar(generate_magnetic_field(param.gs, param.time), param.gs.mu));

    ret = v3d_sub(ret, v3d_scalar(param.gs.ani.dir, 2.0 * param.gs.ani.ani * v3d_dot(param.m, param.gs.ani.dir)));

    return v3d_scalar(ret, -1.0 / param.gs.mu);
}

//@TODO: Add current
v3d dm_dt(parameters param) {
    v3d H_eff = effective_field(param);
    v3d v = v3d_scalar(v3d_cross(param.m, H_eff), -param.gs.gamma);
    return v3d_scalar(v3d_sum(v, v3d_scalar(v3d_cross(param.m, v), param.gs.alpha)), 1.0 / (1.0 + param.gs.alpha * param.gs.alpha));
}

//@TODO: implement RK2 and euler
v3d step(parameters param, double dt) {
    v3d rk1, rk2, rk3, rk4;
    v3d c_ori = param.m;
    double time_ori = param.time;
    rk1 = dm_dt(param);

    param.m = v3d_sum(c_ori, v3d_scalar(rk1, dt / 2.0));
    param.time = time_ori + dt / 2.0;
    rk2 = dm_dt(param);

    param.m = v3d_sum(c_ori, v3d_scalar(rk2, dt / 2.0));
    param.time = time_ori + dt / 2.0;
    rk3 = dm_dt(param);

    param.m = v3d_sum(c_ori, v3d_scalar(rk3, dt));
    param.time = time_ori + dt;
    rk4 = dm_dt(param);

    return v3d_scalar(v3d_sum(v3d_sum(rk1, v3d_scalar(rk2, 2.0)), v3d_sum(v3d_scalar(rk3, 2.0), rk4)), dt / 6.0);
}

/*



double charge_derivative(parameters param);
double charge_lattice(parameters param);
*/
