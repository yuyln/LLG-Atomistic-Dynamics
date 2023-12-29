#include "constants.h"
#include "simulation_funcs.h"
#include <math.h>

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
    //%s
    double normalized = 0.5;
    //normalized += sin(time / 0.1e-11) * 0.2;
    normalized *= gs.dm * gs.dm / gs.exchange;
    double real = normalized / gs.mu;
    return v3d_c(0, 0, -real);
}

current generate_current(grid_site_param gs, double time) {
    UNUSED(gs);
    UNUSED(time);
    //%s
    current ret = {0};
    ret.type = CUR_STT;
    ret.stt.polarization = -1.0;
    ret.stt.beta = 0.0;
    ret.stt.j = v3d_scalar(v3d_c(1, 0, 0), 1.0e11);
    return ret;
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

double cubic_anisotropy_energy(parameters param) {
    return -(param.m.x * param.m.x * param.m.x * param.m.x + 
             param.m.y * param.m.y * param.m.y * param.m.y +
             param.m.z * param.m.z * param.m.z * param.m.z) * param.gs.cubic_ani;
}

double field_energy(parameters param) {
    return -param.gs.mu * v3d_dot(generate_magnetic_field(param.gs, param.time), param.m);
}

double energy(parameters param) {
    return 0.5 * exchange_energy(param) + 0.5 * dm_energy(param) + anisotropy_energy(param) + field_energy(param) + cubic_anisotropy_energy(param);
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

    ret = v3d_sub(ret, v3d_scalar(v3d_c(param.m.x * param.m.x * param.m.x,
                                        param.m.y * param.m.y * param.m.y,
                                        param.m.z * param.m.z * param.m.z), 4.0 * param.gs.cubic_ani));

    return v3d_scalar(ret, -1.0 / param.gs.mu);
}

//@TODO: Add Z derivative
v3d v3d_dot_grad(v3d v, neighbors_set neigh, double dx, double dy) {
    v3d ret = {0};
    ret.x = v.x * (neigh.right.x - neigh.left.x) / (2.0 * dx) +
            v.y * (neigh.up.x - neigh.down.x) / (2.0 * dy);

    ret.y = v.x * (neigh.right.y - neigh.left.y) / (2.0 * dx) +
            v.y * (neigh.up.y - neigh.down.y) / (2.0 * dy);

    ret.z = v.x * (neigh.right.z - neigh.left.z) / (2.0 * dx) +
            v.y * (neigh.up.z - neigh.down.z) / (2.0 * dy);

    return ret;
}

v3d dm_dt(parameters param) {
    v3d H_eff = effective_field(param);
    v3d v = v3d_scalar(v3d_cross(param.m, H_eff), -param.gs.gamma);
    current cur = generate_current(param.gs, param.time);
    switch (cur.type) {
        case CUR_STT: {
            v3d common = v3d_dot_grad(cur.stt.j, param.neigh, param.gs.lattice, param.gs.lattice);
            common = v3d_scalar(common, cur.stt.polarization * param.gs.lattice * param.gs.lattice * param.gs.lattice / (2.0 * QE));
            v3d beta = v3d_scalar(v3d_cross(param.m, common), cur.stt.beta);
            v = v3d_sum(v, v3d_sub(common, beta));
        }
        break;
        case CUR_SHE: {
            v3d common = v3d_scalar(v3d_cross(param.m, cur.she.p), cur.she.theta_sh * param.gs.lattice * param.gs.lattice * param.gs.lattice / (2.0 * cur.she.thickness * QE));
            v3d beta = v3d_scalar(v3d_cross(param.m, common), cur.stt.beta);
            v = v3d_sum(v, v3d_sub(v3d_cross(common, param.m), beta));
        }
        break;
        case CUR_BOTH: {
            v3d stt_common = v3d_dot_grad(cur.stt.j, param.neigh, param.gs.lattice, param.gs.lattice);
            stt_common = v3d_scalar(stt_common, cur.stt.polarization * param.gs.lattice * param.gs.lattice * param.gs.lattice / (2.0 * QE));
            v3d stt_beta = v3d_scalar(v3d_cross(param.m, stt_common), cur.stt.beta);
            v = v3d_sum(v, v3d_sub(stt_common, stt_beta));


            v3d she_common = v3d_scalar(v3d_cross(param.m, cur.she.p), cur.she.theta_sh * param.gs.lattice * param.gs.lattice * param.gs.lattice / (2.0 * cur.she.thickness * QE));
            v3d she_beta = v3d_scalar(v3d_cross(param.m, she_common), cur.stt.beta);
            v = v3d_sum(v, v3d_sub(v3d_cross(she_common, param.m), she_beta));
        }
        break;
        case CUR_NONE:
        break;
    }

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

double charge_derivative(v3d m, v3d left, v3d right, v3d up, v3d down) {
    return v3d_dot(m, v3d_cross(
                v3d_scalar(v3d_sub(right, left), 0.5), //x derivative scaled by lattice
                v3d_scalar(v3d_sub(up, down), 0.5)  //y derivative scaled by lattice
                )) * 1.0 / (4.0 * M_PI);
}


//https://iopscience.iop.org/article/10.1088/2633-1357/abad0c/pdf
double q_ijk(v3d mi, v3d mj, v3d mk) {
    double num = v3d_dot(mi, v3d_cross(mj, mk));
    double den = 1.0 + v3d_dot(mi, mj) + v3d_dot(mi, mk) + v3d_dot(mj, mk);
    return 2.0 * atan2(num, den);
}

/*
    (__)---(m2)---(__)
     |      |      |
     |      |      |
    (m3)---(m0)---(m1)
     |      |      |
     |      |      |
    (__)---(m4)---(__)
*/

double charge_lattice(v3d m, v3d left, v3d right, v3d up, v3d down) {
    double q_012 = q_ijk(m, right, up);
    double q_023 = q_ijk(m, up, left);
    double q_034 = q_ijk(m, left, down);
    double q_041 = q_ijk(m, down, right);
    return 1.0 / (8.0 * M_PI) * (q_012 + q_023 + q_034 + q_041);
}

v3d emergent_magnetic_field_lattice(v3d m, v3d left, v3d right, v3d up, v3d down) {
    return v3d_c(0.0, 0.0, HBAR / QE * charge_lattice(m, left, right, up, down));
}

v3d emergent_magnetic_field_derivative(v3d m, v3d left, v3d right, v3d up, v3d down) {
    return v3d_c(0.0, 0.0, HBAR / QE * charge_derivative(m, left, right, up, down));
}

v3d emergent_eletric_field(v3d m, v3d left, v3d right, v3d up, v3d down, v3d dmdt, double dx, double dy) {
    return v3d_c(
            HBAR / QE * v3d_dot(m, v3d_cross(v3d_scalar(v3d_sub(right, left), 1.0 / (2.0 * dx)), dmdt)),
            HBAR / QE * v3d_dot(m, v3d_cross(v3d_scalar(v3d_sub(up, down), 1.0 / (2.0 * dy)), dmdt)),
            0.0
            );
}
