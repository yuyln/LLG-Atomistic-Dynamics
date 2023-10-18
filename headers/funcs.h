#ifndef __FUNCS
#define __FUNCS

#ifndef OPENCLCOMP
#include "grid.h"
#include "constants.h"
#endif


current_t generate_current(int i, int j, grid_param_t gp, current_t base, double norm_time) {
    UNUSED(i); UNUSED(j); UNUSED(gp); UNUSED(norm_time);
    return base;
}

v3d generate_field(int i, int j, grid_param_t gp, v3d base, double norm_time) {
    UNUSED(i); UNUSED(j); UNUSED(gp); UNUSED(norm_time);
    return base;
}

v3d linear_interpolation(v3d v1, v3d v2, double t) {
    return v3d_add(v1, v3d_scalar(v3d_sub(v2, v1), t));
}

v3d bilinear_interpolation(v3d v00, v3d v10, v3d v11, v3d v01, double u, double v) {
    v3d b = v3d_sub(v10, v00);
    v3d c = v3d_sub(v01, v00);
    v3d d = v3d_sub(v11, v10);
    d = v3d_sub(d, v01);
    d = v3d_add(d, v00);

    v3d ret = v00;
    ret = v3d_add(ret, v3d_scalar(b, u));
    ret = v3d_add(ret, v3d_scalar(c, v));
    ret = v3d_add(ret, v3d_scalar(d, u * v));
    return ret;
}

//WARNING: This should be the ONLY access to global memory when running on GPU
v3d get_pbc_v3d(int row, int col, const GLOBAL v3d* v, int rows, int cols, pbc_t pbc) {
    switch (pbc.pbc_type) {
        case PBC_NONE: {
            if (row >= rows || row < 0 || col >= cols || col < 0)
                return pbc.dir;
            break;
        }
        case PBC_X: {
            if (row >= rows || row < 0)
                return pbc.dir;
            col = ((col % cols) + cols) % cols;
            break;
        }
        case PBC_Y: {
            if (col >= cols || col < 0)
                return pbc.dir;
            row = ((row % rows) + rows) % rows;
            break;
        }
        case PBC_XY: {
            col = ((col % cols) + cols) % cols;
            row = ((row % rows) + rows) % rows;
            break;
        }
    }
    return v[row * cols + col];
}

//TODO: Is options for more than first neighbours needed?
v3d get_dm_v3d(int drow, int dcol, DM_TYPE dm_type, double dm) {
    switch (dm_type) {
        case R_ij:
            return v3d_c(dcol * dm, drow * dm, 0.0);

        /*case R_ij_ANISOTROPIC_X:
            return v3d_c(-dcol * dm, drow * dm, 0.0);

        case R_ij_ANISOTROPIC_Y:
            return v3d_c(dcol * dm, -drow * dm, 0.0);*/

        case Z_CROSS_R_ij:
            return v3d_c(-drow * dm, dcol * dm, 0.0);

        /*case Z_CROSS_R_ij_ANISOTROPIC_X:
            return v3d_c(drow * dm, dcol * dm, 0.0);

        case Z_CROSS_R_ij_ANISOTROPIC_Y:
            return v3d_c(-drow * dm, -dcol * dm, 0.0);*/

        case R_ij_CROSS_Z:
            return v3d_c(drow * dm, -dcol * dm, 0.0);
    }

    return v3d_s(0.0);
}

double exchange_energy(v3d c, v3d left, v3d right, v3d up, v3d down,
                       grid_param_t gp, region_param_t region) {
    return -gp.exchange * region.exchange_mult * (v3d_dot(c, right)+
                                                 v3d_dot(c, left) +
                                                 v3d_dot(c, up)   +
                                                 v3d_dot(c, down));
}

double dm_energy(v3d c, v3d left, v3d right, v3d up, v3d down,
                 grid_param_t gp, region_param_t region) {

    v3d DMR = get_dm_v3d(0, 1, region.dm_type, (gp.dm + gp.dm_ani) * region.dm_mult), //put (-) here to remove from later
        DML = get_dm_v3d(0, -1, region.dm_type, (gp.dm + gp.dm_ani) * region.dm_mult),
        DMU = get_dm_v3d(1, 0, region.dm_type, (gp.dm - gp.dm_ani) * region.dm_mult),
        DMD = get_dm_v3d(-1, 0, region.dm_type, (gp.dm - gp.dm_ani) * region.dm_mult);

    return -(v3d_dot(DMR, v3d_cross(c, right))+
             v3d_dot(DML, v3d_cross(c, left)) +
             v3d_dot(DMU, v3d_cross(c, up))   +
             v3d_dot(DMD, v3d_cross(c, down)));

}

double zeeman_energy(int row, int col, v3d c, grid_param_t gp, v3d field) {
    return -v3d_dot(c, generate_field(row, col, gp, field, 0.0));
}

double anisotropy_energy(v3d c, anisotropy_t ani) {
    return -ani.K_1 * (v3d_dot(c, ani.dir)) * (v3d_dot(c, ani.dir));
}

double cubic_anisotropy_energy(v3d c, grid_param_t gp) {
    return -gp.cubic_ani * (c.x * c.x * c.x * c.x+
                            c.y * c.y * c.y * c.y+
                            c.z * c.z * c.z * c.z);
}

double hamiltonian_I(int row, int col,
                     v3d c, v3d left, v3d right, v3d up, v3d down,
                     grid_param_t gp, anisotropy_t ani, region_param_t region, v3d field) {


    double out = zeeman_energy(row, col, c, gp, field);

    out += 0.5 * exchange_energy(c, left, right, up, down, gp, region);

    out += 0.5 * dm_energy(c, left, right, up, down, gp, region);

    out += anisotropy_energy(c, ani);
    out += cubic_anisotropy_energy(c, gp);

    return out;
}

double hamiltonian(GLOBAL grid_t* g, v3d field) CPU_ONLY { 
    double ret = 0.0;
    for (uint64_t I = 0; I < g->param.total; ++I) {
        int col = I % g->param.cols;
        int row = (I - col) / g->param.cols;
        v3d c = get_pbc_v3d(row, col, g->grid, g->param.rows, g->param.cols, g->param.pbc);
        v3d l = get_pbc_v3d(row, col - 1, g->grid, g->param.rows, g->param.cols, g->param.pbc);
        v3d r = get_pbc_v3d(row, col + 1, g->grid, g->param.rows, g->param.cols, g->param.pbc);
        v3d u = get_pbc_v3d(row + 1, col, g->grid, g->param.rows, g->param.cols, g->param.pbc);
        v3d d = get_pbc_v3d(row - 1, col, g->grid, g->param.rows, g->param.cols, g->param.pbc);
        ret += hamiltonian_I(row, col, c, l, r, u, d, g->param, g->ani[I], g->regions[I], field);
    }
    return ret;
}

v3d grid_normalize(v3d v, pinning_t pin) {
    if (pin.fixed)
        v = pin.dir;
    else
        v = v3d_normalize(v);
    return v;
}

v3d v3d_dot_grad_v3d(v3d v, v3d left, v3d right, v3d up, v3d down, double dx, double dy) {
    v3d ret;
    
    ret.x = v.x * (right.x - left.x) / (2.0 * dx)+
            v.y * (up.x - down.x) / (2.0 * dy);

    ret.y = v.x * (right.y - left.y) / (2.0 * dx)+
            v.y * (up.y - down.y) / (2.0 * dy);
    
    ret.z = v.x * (right.z - left.z) / (2.0 * dx)+
            v.y * (up.z - down.z) / (2.0 * dy);
    
    return ret;
}

v3d dH_dSi(int row, int col,
           v3d c, v3d left, v3d right, v3d up, v3d down, 
           grid_param_t gp, region_param_t region, anisotropy_t ani, v3d field, double norm_time) {
    v3d ret = v3d_s(0.0);

    v3d DMR = get_dm_v3d(0, 1, region.dm_type, -(gp.dm + gp.dm_ani) * region.dm_mult), //put (-) here to remove from later
        DML = get_dm_v3d(0, -1, region.dm_type, -(gp.dm + gp.dm_ani) * region.dm_mult),
        DMU = get_dm_v3d(1, 0, region.dm_type, -(gp.dm - gp.dm_ani) * region.dm_mult),
        DMD = get_dm_v3d(-1, 0, region.dm_type, -(gp.dm - gp.dm_ani) * region.dm_mult);
    
    v3d exchange = v3d_scalar(right, -gp.exchange * region.exchange_mult);
    exchange = v3d_add(exchange, v3d_scalar(left, -gp.exchange * region.exchange_mult));
    exchange = v3d_add(exchange, v3d_scalar(up, -gp.exchange * region.exchange_mult));
    exchange = v3d_add(exchange, v3d_scalar(down, -gp.exchange * region.exchange_mult));
    ret = v3d_add(ret, exchange);

    v3d dm = v3d_cross(right, DMR);
    dm = v3d_add(dm, v3d_cross(left, DML));
    dm = v3d_add(dm, v3d_cross(up, DMU));
    dm = v3d_add(dm, v3d_cross(down, DMD));
    ret = v3d_add(ret, dm);

    v3d ax_ani = v3d_scalar(ani.dir, -2.0 * ani.K_1 * v3d_dot(c, ani.dir));
    ret = v3d_add(ret, ax_ani);

    v3d cub_ani = v3d_c(-4.0 * gp.cubic_ani * c.x * c.x * c.x,
                        -4.0 * gp.cubic_ani * c.y * c.y * c.y,
                        -4.0 * gp.cubic_ani * c.z * c.z * c.z);
    ret = v3d_add(ret, cub_ani);
    
    ret = v3d_sub(ret, generate_field(row, col, gp, field, norm_time));

    return ret;
}

v3d ds_dtau(int row, int col,
            v3d c, v3d left, v3d right, v3d up, v3d down, 
            grid_param_t gp, region_param_t region, anisotropy_t ani,
            v3d field, current_t cur, double norm_time) {
    v3d Heff = dH_dSi(row, col, c, left, right, up, down, gp, region, ani, field, norm_time);
    v3d V = v3d_cross(c, Heff);

    switch (cur.type) {
        case CUR_CPP: {
            cur = generate_current(row, col, gp, cur, norm_time);
            double factor = cur.theta_sh * gp.lattice / cur.thick;
            v3d cur_local = v3d_scalar(v3d_cross(cur.p, c), factor);
            V = v3d_add(V, v3d_cross(c, cur_local));
            V = v3d_add(V, v3d_scalar(cur_local, -cur.beta));
            break;
        }
        case CUR_STT: {
            cur = generate_current(row, col, gp, cur, norm_time);
            v3d cur_local = v3d_dot_grad_v3d(cur.j, left, right, up, down, 1.0, 1.0);
            V = v3d_add(V, v3d_scalar(cur_local, cur.P));
            V = v3d_sub(V, v3d_scalar(v3d_cross(c, cur_local), cur.P * cur.beta));
            break;
        }
        case CUR_BOTH: {

            cur = generate_current(row, col, gp, cur, norm_time);
            double factor = cur.theta_sh * gp.lattice / cur.thick;
            v3d cur_local = v3d_scalar(v3d_cross(cur.p, c), factor);
            V = v3d_add(V, v3d_cross(c, cur_local));
            V = v3d_add(V, v3d_scalar(cur_local, -cur.beta));


            cur = generate_current(row, col, gp, cur, norm_time);
            cur_local = v3d_dot_grad_v3d(cur.j, left, right, up, down, 1.0, 1.0);
            V = v3d_add(V, v3d_scalar(cur_local, cur.P));
            V = v3d_sub(V, v3d_scalar(v3d_cross(c, cur_local), cur.P * cur.beta));

            break;
        }
        case CUR_NONE:
            break;
        }

    V = v3d_add(V, v3d_scalar(v3d_cross(c, V), gp.alpha));
    return v3d_scalar(V, 1.0 / (1.0 + gp.alpha * gp.alpha));
}

v3d grid_step(int row, int col, 
              v3d c, v3d left, v3d right, v3d up, v3d down,
              grid_param_t gp, region_param_t region, anisotropy_t ani,
              v3d field, current_t cur, double dt, double norm_time) {
    #if defined(RK4)
    v3d rk1, rk2, rk3, rk4;
    rk1 = ds_dtau(row, col, c, left, right, up, down, gp, region, ani, field, cur, norm_time);

    rk2 = ds_dtau(row, col, v3d_add(c, v3d_scalar(rk1, dt / 2.0)), left, right, up, down, gp, region, ani, field, cur, norm_time + dt / 2.0);

    rk3 = ds_dtau(row, col, v3d_add(c, v3d_scalar(rk2, dt / 2.0)), left, right, up, down, gp, region, ani, field, cur, norm_time + dt / 2.0);

    rk4 = ds_dtau(row, col, v3d_add(c, v3d_scalar(rk3, dt)), left, right, up, down, gp, region, ani, field, cur, norm_time + dt);

    return v3d_scalar(v3d_add(v3d_add(rk1, v3d_scalar(rk2, 2.0)), v3d_add(v3d_scalar(rk3, 2.0), rk4)), dt / 6.0);

    #elif defined(RK2)
    v3d rk1, rk2;

    rk1 = ds_dtau(row, col, c, left, right, up, down, gp, region, ani, field, cur, norm_time);

    rk2 = ds_dtau(row, col, v3d_add(c, v3d_scalar(rk1, dt)), left, right, up, down, gp, region, ani, field, cur, norm_time + dt);

    return v3d_scalar(v3d_add(rk1, rk2), dt / 2.0);

    #elif defined(EULER)
    return ds_dtau(row, col, c, left, right, up, down, gp, region, ani, field, cur, norm_time);

    #else
    return (v3d){0.0, 0.0, 0.0};

    #endif
}

double charge_old(v3d c, v3d left, v3d right, v3d up, v3d down) {
    v3d dgdx = v3d_scalar(v3d_sub(right, left), 0.5);
    v3d dgdy = v3d_scalar(v3d_sub(up, down), 0.5);
    return 1.0 / (4 * M_PI) * v3d_dot(c, v3d_cross(dgdx, dgdy));
}

double Q_ijk(v3d mi, v3d mj, v3d mk) {
    double num = v3d_dot(mi, v3d_cross(mj, mk));
    double den = 1.0 + v3d_dot(mi, mj) + v3d_dot(mi, mk) + v3d_dot(mj, mk);
    return 2.0 * atan2(num, den);
}

double charge(v3d c, v3d left, v3d right, v3d up, v3d down) {
     //https://iopscience.iop.org/article/10.1088/2633-1357/abad0c/pdf

    v3d m2 = right,//get_pbc_v3d(row, col + 1, g, rows, cols, pbc),
        m3 = up,//get_pbc_v3d(row + 1, col, g, rows, cols, pbc),
        m4 = left,//get_pbc_v3d(row, col - 1, g, rows, cols, pbc),
        m5 = down,//get_pbc_v3d(row - 1, col, g, rows, cols, pbc),
        m1 = c;//get_pbc_v3d(row, col, g, rows, cols, pbc);

    double q_123 = Q_ijk(m1, m2, m3);
    double q_134 = Q_ijk(m1, m3, m4);
    double q_145 = Q_ijk(m1, m4, m5);
    double q_152 = Q_ijk(m1, m5, m2);

    return 1.0 / (8.0 * M_PI) * (q_123 + q_134 + q_145 + q_152);
}


v3d B_emergent(v3d c, v3d left, v3d right, v3d up, v3d down) {
    return v3d_c(0.0, 0.0, 4 * M_PI * charge_old(c, left, right, up, down));
}

//0->before
//1->current
//2->after
v3d E_emergent(v3d c0, v3d c1, v3d c2, v3d left1, v3d right1, v3d up1, v3d down1, double dt) {
    v3d ret = v3d_s(0.0);
    
    v3d dgdx = v3d_scalar(v3d_sub(right1, left1), 0.5);
    v3d dgdy = v3d_scalar(v3d_sub(up1, down1), 0.5);
    v3d dgdt = v3d_scalar(v3d_sub(c2, c0), 0.5 / dt);
    ret.x = v3d_dot(c1, v3d_cross(dgdx, dgdt));
    ret.y = v3d_dot(c1, v3d_cross(dgdy, dgdt));
    return ret;
}

v3d velocity(v3d c0, v3d c1, v3d c2, v3d left1, v3d right1, v3d up1, v3d down1, double dt) {
    v3d Em = E_emergent(c0, c1, c2, left1, right1, up1, down1, dt);
    v3d Bm = B_emergent(c1, left1, right1, up1, down1);
    return (v3d){ Em.y / Bm.z, -Em.x / Bm.z, 0.0 };
}

v3d velocity_weighted(v3d c0, v3d c1, v3d c2, v3d left1, v3d right1, v3d up1, v3d down1, double dt) {
    v3d Em = E_emergent(c0, c1, c2, left1, right1, up1, down1, dt);
    double factor = 1.0 / (4.0 * M_PI);
    return (v3d){ factor * Em.y, -factor * Em.x, 0.0 };
}

v3d gradient_descente_velocity(v3d g_p, v3d g_n, double dt) {
    return v3d_scalar(
		    v3d_sub(g_n, g_p),
		    1.0 / (2.0 * dt)
		    );
}

v3d gradient_descent_force(int row, int col,
                           v3d c, v3d left, v3d right, v3d up, v3d down,
                           grid_param_t gp, region_param_t region, anisotropy_t ani,
                           v3d field, double alpha, double beta, v3d vel) {
    v3d F = dH_dSi(row, col, c, left, right, up, down, gp, region, ani, field, 0);
    F = v3d_sub(F, v3d_scalar(vel, alpha));
    F = v3d_sub(F, v3d_scalar(c, beta));
    return F;
}
#endif
