#include "simulation_funcs.h"

v3d apply_pbc(GLOBAL v3d *v, matrix_size size, matrix_loc loc, pbc_rules pbc) {
    int pbc_x = pbc.dirs & (1 << 0);
    int pbc_y = pbc.dirs & (1 << 1);
    int pbc_z = pbc.dirs & (1 << 2);

    if (loc.dim[0] >= size.dim[0] || loc.dim[0] < 0) {
        if (!pbc_y)
            return pbc.m;
        loc.dim[0] = ((loc.dim[0] % size.dim[0]) + size.dim[0]) % size.dim[0];
    }

    if (loc.dim[1] >= size.dim[1] || loc.dim[1] < 0) {
        if (!pbc_x)
            return pbc.m;
        loc.dim[1] = ((loc.dim[1] % size.dim[1]) + size.dim[1]) % size.dim[1];
    }

#ifndef NBULK
    if (loc.dim[2] >= size.dim[2] || loc.dim[2] < 0) {
        if (!pbc_z)
            return pbc.m;
        loc.dim[2] = ((loc.dim[2] % size.dim[2]) + size.dim[2]) % size.dim[2];
    }
#endif

    return v[LOC(loc.row, loc.col, loc.depth, size.rows, size.cols)];
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
    return -(v3d_dot(param.c, param.l) + v3d_dot(param.c, param.r) +
             v3d_dot(param.c, param.u) + v3d_dot(param.c, param.d)
#ifndef NBULK
            +v3d_dot(param.c, param.b) + v3d_dot(param.c, param.f)
#endif
             ) * param.gs.exchange;
}

double dm_energy(parameters param) {
    double ret = 0.0;
    ret += v3d_dot(get_dm_vec(v3d_c(1, 0, 0), param.gs.dm + param.gs.dm_ani, param.gs.dm_sym), v3d_cross(param.c, param.r));
    ret += v3d_dot(get_dm_vec(v3d_c(-1, 0, 0), param.gs.dm + param.gs.dm_ani, param.gs.dm_sym), v3d_cross(param.c, param.l));
    ret += v3d_dot(get_dm_vec(v3d_c(0, 1, 0), param.gs.dm - param.gs.dm_ani, param.gs.dm_sym), v3d_cross(param.c, param.u));
    ret += v3d_dot(get_dm_vec(v3d_c(0, -1, 0), param.gs.dm - param.gs.dm_ani, param.gs.dm_sym), v3d_cross(param.c, param.d));
#ifndef NBULK
    ret += v3d_dot(get_dm_vec(v3d_c(0, 0, 1), param.gs.dm + param.gs.dm_ani, param.gs.dm_sym), v3d_cross(param.c, param.f));
    ret += v3d_dot(get_dm_vec(v3d_c(0, 0, -1), param.gs.dm + param.gs.dm_ani, param.gs.dm_sym), v3d_cross(param.c, param.b));
#endif
    return -ret;
}

double anisotropy_energy(parameters param) {
    double scalar = v3d_dot(param.c, param.gs.ani.dir);
    return -param.gs.ani.ani * scalar * scalar;
}

//double cubic_anisotropy_energy(parameters param);

double field_energy(parameters param) {
    return -param.gs.mu * v3d_dot(generate_magnetic_field(param.gs, param.time), param.c);
}

double energy(parameters param) {
    return 0.5 * exchange_energy(param) + 0.5 * dm_energy(param) + anisotropy_energy(param) + field_energy(param) /*+ cubic_anisotropy_energy(param)*/;
}

v3d effective_field(parameters param) {
    v3d ret = v3d_s(0);

    v3d exchange_field = v3d_sum(v3d_sum(param.r, param.l), v3d_sum(param.u, param.d));
#ifndef NBULK
    exchange_field = v3d_sum(exchange_field, v3d_sum(param.f, param.b));
#endif
    exchange_field = v3d_scalar(exchange_field, param.gs.exchange);
    ret = v3d_sub(ret, exchange_field);


    v3d dm_field_x = v3d_sum(v3d_cross(get_dm_vec(v3d_c(1, 0, 0), param.gs.dm + param.gs.dm_ani, param.gs.dm_sym), param.r),
                             v3d_cross(get_dm_vec(v3d_c(-1, 0, 0), param.gs.dm + param.gs.dm_ani, param.gs.dm_sym), param.l));

    v3d dm_field_y = v3d_sum(v3d_cross(get_dm_vec(v3d_c(0, 1, 0), param.gs.dm - param.gs.dm_ani, param.gs.dm_sym), param.u),
                             v3d_cross(get_dm_vec(v3d_c(0, -1, 0), param.gs.dm - param.gs.dm_ani, param.gs.dm_sym), param.d));
    v3d dm_field = v3d_sum(dm_field_x, dm_field_y);
#ifndef NBULK
    v3d dm_field_z = v3d_sum(v3d_cross(get_dm_vec(v3d_c(0, 0, 1), param.gs.dm + param.gs.dm_ani, param.gs.dm_sym), param.f),
                             v3d_cross(get_dm_vec(v3d_c(0, 0, -1), param.gs.dm + param.gs.dm_ani, param.gs.dm_sym), param.b));
    dm_field = v3d_sum(dm_field, dm_field_z);
#endif
    ret = v3d_sub(ret, dm_field);

    ret = v3d_sub(ret, v3d_scalar(generate_magnetic_field(param.gs, param.time), param.gs.mu));

    ret = v3d_sub(ret, v3d_scalar(param.gs.ani.dir, 2.0 * param.gs.ani.ani * v3d_dot(param.c, param.gs.ani.dir)));

    return v3d_scalar(ret, -1.0 / param.gs.mu);
}

//@TODO: Add current
v3d dm_dt(parameters param) {
    v3d H_eff = effective_field(param);
    v3d v = v3d_scalar(v3d_cross(param.c, H_eff), -param.gs.gamma);
    return v3d_scalar(v3d_sum(v, v3d_scalar(v3d_cross(param.c, v), param.gs.alpha)), 1.0 / (1.0 + param.gs.alpha * param.gs.alpha));
}

//@TODO: implement RK2 and euler
v3d step(parameters param, double dt) {
    v3d rk1, rk2, rk3, rk4;
    v3d c_ori = param.c;
    double time_ori = param.time;
    rk1 = dm_dt(param);

    param.c = v3d_sum(c_ori, v3d_scalar(rk1, dt / 2.0));
    param.time = time_ori + dt / 2.0;
    rk2 = dm_dt(param);

    param.c = v3d_sum(c_ori, v3d_scalar(rk2, dt / 2.0));
    param.time = time_ori + dt / 2.0;
    rk3 = dm_dt(param);

    param.c = v3d_sum(c_ori, v3d_scalar(rk3, dt));
    param.time = time_ori + dt;
    rk4 = dm_dt(param);

    return v3d_scalar(v3d_sum(v3d_sum(rk1, v3d_scalar(rk2, 2.0)), v3d_sum(v3d_scalar(rk3, 2.0), rk4)), dt / 6.0);
}

/*



double charge_derivative(parameters param);
double charge_lattice(parameters param);
*/
