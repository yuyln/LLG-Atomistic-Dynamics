#ifndef __FUNCS
#define __FUNCS

#ifndef OPENCLCOMP
#include "./headers/grid.h"
#include "./headers/constants.h"
#endif

current_t generate_current(size_t I, GLOBAL grid_param_t* g, current_t base, double norm_time) {
    UNUSED(I); UNUSED(g); UNUSED(norm_time);
    return base;
}

v3d generate_field(size_t I, GLOBAL grid_param_t* g, v3d base, double norm_time) {
    UNUSED(I); UNUSED(g); UNUSED(norm_time);
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

v3d get_pbc_v3d(int row, int col, const GLOBAL v3d* v, int rows, int cols, pbc_t pbc) {
    switch (pbc.pbc_type) {
    case pbc_t_NONE:
        if (row >= rows || row < 0 || col >= cols || col < 0)
            return pbc.dir;
        break;
    
    case pbc_t_X:
        if (row >= rows || row < 0)
            return pbc.dir;
        if (col >= cols)
            col = col % cols;
        else if (col < 0)
            col = (col * (1 - cols)) % cols;
        break;

    case pbc_t_Y:
        if (col >= cols || col < 0)
            return pbc.dir;
        
        if (row >= rows)
            row = row % rows;
        else if (row < 0)
            row = (row * (1 - rows)) % rows;
        break;
    
    case pbc_t_XY:
        if (col >= cols)
            col = col % cols;
        else if (col < 0)
            col = (col * (1 - cols)) % cols;
        if (row >= rows)
            row = row % rows;
        else if (row < 0)
            row = (row * (1 - rows)) % rows;
    }
    return v[row * cols + col];
}

v3d get_dm_v3d(int drow, int dcol, DM_TYPE dm_type, double dm) {
    switch (dm_type) {
    case R_ij:
        if (drow * drow + dcol * dcol > 1)
            return v3d_normalize_to(v3d_c(dcol, drow, 0), dm);
        return v3d_c(dcol * dm, drow * dm, 0.0);

    case Z_CROSS_R_ij:
        if (drow * drow + dcol * dcol > 1)
            return v3d_normalize_to(v3d_c(-drow, dcol, 0), dm);
        return v3d_c(-drow * dm, dcol * dm, 0.0);
    }
    return v3d_s(0.0);
}

//double hamiltonian_I(size_t I, GLOBAL grid_t* g, v3d field)
double hamiltonian_I(size_t I, GLOBAL v3d *v, GLOBAL grid_param_t *param, GLOBAL anisotropy_t *anis, GLOBAL region_param_t *regions, v3d field) {
    int col = I % param->cols;
    int row = (I - col) / param->cols;
    v3d C = get_pbc_v3d(row, col, v, param->rows, param->cols, param->pbc),
        R = get_pbc_v3d(row, col + 1, v, param->rows, param->cols, param->pbc),
        L = get_pbc_v3d(row, col - 1, v, param->rows, param->cols, param->pbc),
        U = get_pbc_v3d(row + 1, col, v, param->rows, param->cols, param->pbc),
        D = get_pbc_v3d(row - 1, col, v, param->rows, param->cols, param->pbc);

    v3d DMR = get_dm_v3d(0, 1, regions[I].dm_type, param->dm * regions[I].dm_mult),
        DML = get_dm_v3d(0, -1, regions[I].dm_type, param->dm * regions[I].dm_mult),
        DMU = get_dm_v3d(1, 0, regions[I].dm_type, param->dm * regions[I].dm_mult),
        DMD = get_dm_v3d(-1, 0, regions[I].dm_type, param->dm * regions[I].dm_mult);

    double out = -param->mu_s * v3d_dot(C, generate_field(I, param, field, 0.0)) * regions[I].field_mult;


    out += -0.5 * param->exchange * regions[I].exchange_mult * (v3d_dot(C, R)+
                                       v3d_dot(C, L)+
                                       v3d_dot(C, U)+
                                       v3d_dot(C, D));

    out += -0.5 * (v3d_dot(DMR, v3d_cross(C, R))+
                   v3d_dot(DML, v3d_cross(C, L))+
                   v3d_dot(DMU, v3d_cross(C, U))+
                   v3d_dot(DMD, v3d_cross(C, D)));

    out += -anis[I].K_1 * (v3d_dot(C, anis[I].dir)) * (v3d_dot(C, anis[I].dir));

    out += -param->cubic_ani * (C.x * C.x * C.x * C.x+
                                  C.y * C.y * C.y * C.y+
                                  C.z * C.z * C.z * C.z);

    return out;
}

double hamiltonian(GLOBAL grid_t* g, v3d field) {
    double ret = 0.0;
    for (size_t I = 0; I < g->param.total; ++I)
        ret += hamiltonian_I(I, g->grid, &g->param, g->ani, g->regions, field);
    return ret;
}

void grid_normalize(size_t I, GLOBAL v3d* v, GLOBAL pinning_t *pin) {
    if (pin[I].fixed)
        v[I] = pin[I].dir;
    else
        v[I] = v3d_normalize(v[I]);
}

v3d v3d_dot_grad_v3d(size_t I, v3d v, GLOBAL v3d* g, int rows, int cols, double dx, double dy, pbc_t pbc) {
    v3d ret;
    int col = I % cols;
    int row = (I - col) / cols;
    v3d R = get_pbc_v3d(row, col + 1, g, rows, cols, pbc),
        L = get_pbc_v3d(row, col - 1, g, rows, cols, pbc),
        U = get_pbc_v3d(row + 1, col, g, rows, cols, pbc),
        D = get_pbc_v3d(row - 1, col, g, rows, cols, pbc);
    
    ret.x = v.x * (R.x - L.x) / (2.0 * dx)+
            v.y * (U.x - D.x) / (2.0 * dy);

    ret.y = v.x * (R.y - L.y) / (2.0 * dx)+
            v.y * (U.y - D.y) / (2.0 * dy);
    
    ret.z = v.x * (R.z - L.z) / (2.0 * dx)+
            v.y * (U.z - D.z) / (2.0 * dy);
    
    return ret;
}

//v3d dH_dSi(size_t I, v3d C, GLOBAL grid_t* g, v3d field, double norm_time)
v3d dH_dSi(size_t I, v3d C, GLOBAL v3d *v, GLOBAL grid_param_t *param, GLOBAL region_param_t *regions, GLOBAL anisotropy_t *anis, v3d field, double norm_time) {
    int col = I % param->cols;
    int row = (I - col) / param->cols;
    v3d ret = v3d_s(0.0);

    v3d R = get_pbc_v3d(row, col + 1, v, param->rows, param->cols, param->pbc),
        L = get_pbc_v3d(row, col - 1, v, param->rows, param->cols, param->pbc),
        U = get_pbc_v3d(row + 1, col, v, param->rows, param->cols, param->pbc),
        D = get_pbc_v3d(row - 1, col, v, param->rows, param->cols, param->pbc);
    
    v3d DMR = get_dm_v3d(0, 1, regions[I].dm_type, -param->dm * regions[I].dm_mult), //put (-) here to remove from later
        DML = get_dm_v3d(0, -1, regions[I].dm_type, -param->dm * regions[I].dm_mult),
        DMU = get_dm_v3d(1, 0, regions[I].dm_type, -param->dm * regions[I].dm_mult),
        DMD = get_dm_v3d(-1, 0, regions[I].dm_type, -param->dm * regions[I].dm_mult);
    
    v3d exchange = v3d_scalar(R, -param->exchange * regions[I].exchange_mult);
    exchange = v3d_add(exchange, v3d_scalar(L, -param->exchange * regions[I].exchange_mult));
    exchange = v3d_add(exchange, v3d_scalar(U, -param->exchange * regions[I].exchange_mult));
    exchange = v3d_add(exchange, v3d_scalar(D, -param->exchange * regions[I].exchange_mult));
    ret = v3d_add(ret, exchange);

    v3d dm = v3d_cross(R, DMR);
    dm = v3d_add(dm, v3d_cross(L, DML));
    dm = v3d_add(dm, v3d_cross(U, DMU));
    dm = v3d_add(dm, v3d_cross(D, DMD));
    ret = v3d_add(ret, dm);

    v3d ani = v3d_scalar(anis[I].dir, -2.0 * anis[I].K_1 * v3d_dot(C, anis[I].dir));
    ret = v3d_add(ret, ani);
    v3d cub_ani = v3d_c(-4.0 * param->cubic_ani * C.x * C.x * C.x,
                          -4.0 * param->cubic_ani * C.y * C.y * C.y,
                          -4.0 * param->cubic_ani * C.z * C.z * C.z);
    ret = v3d_add(ret, cub_ani);
    
    ret = v3d_sub(ret, v3d_scalar(generate_field(I, param, field, norm_time), param->mu_s * regions[I].field_mult));

    return ret;
}

//TODO: Use SHE current, like on Zhang papers
v3d ds_dtau(size_t I, GLOBAL grid_t* g, v3d field, v3d dS, current_t cur, double norm_time) {
    v3d S = v3d_add(g->grid[I], dS);
    v3d Heff = v3d_scalar(dH_dSi(I, S, g->grid, &g->param, g->regions, g->ani, field, norm_time), -1.0 / g->param.mu_s);
    //v3d Heff = v3d_scalar(dH_dSi(I, S, g, field, norm_time), -1.0 / g->param.mu_s);
    double J_abs = g->param.exchange * g->regions[I].exchange_mult * (g->param.exchange * g->regions[I].exchange_mult < 0? -1.0: 1.0);

    v3d V = v3d_scalar(v3d_cross(S, Heff), -g->param.gamma * HBAR / J_abs);

    switch (cur.type) {
    case CUR_CPP: {
        cur = generate_current(I, &g->param, cur, norm_time);
        double factor = g->param.gamma * HBAR * cur.p * g->param.lattice * g->param.avg_spin / (cur.thick * g->param.mu_s);
        v3d cur_local = v3d_scalar(v3d_cross(cur.j, S), factor);
        V = v3d_add(V, v3d_cross(S, cur_local));
        V = v3d_add(V, v3d_scalar(cur_local, cur.beta));
        break;
    }
    case CUR_STT: {
        cur = generate_current(I, &g->param, cur, norm_time);
        v3d cur_local = v3d_dot_grad_v3d(I, cur.j, g->grid, g->param.rows, g->param.cols, g->param.lattice, g->param.lattice, g->param.pbc);
        V = v3d_add(V, v3d_scalar(cur_local, cur.p * g->param.lattice));
        V = v3d_sub(V, v3d_scalar(v3d_cross(S, cur_local), cur.p * cur.beta * g->param.lattice / g->param.avg_spin));
        break;
    }
    case CUR_BOTH: {
        cur = generate_current(I, &g->param, cur, norm_time);
        double factor = g->param.gamma * HBAR * cur.p * g->param.lattice * g->param.avg_spin / (cur.thick * g->param.mu_s);
        v3d cur_local = v3d_scalar(v3d_cross(cur.j, S), factor);
        V = v3d_add(V, v3d_cross(S, cur_local));
        V = v3d_add(V, v3d_scalar(cur_local, cur.beta));

        cur_local = v3d_dot_grad_v3d(I, cur.j, g->grid, g->param.rows, g->param.cols, g->param.lattice, g->param.lattice, g->param.pbc);
        V = v3d_add(V, v3d_scalar(cur_local, cur.p * g->param.lattice));
        V = v3d_sub(V, v3d_scalar(v3d_cross(S, cur_local), cur.p * cur.beta * g->param.lattice / g->param.avg_spin));
        break;
    }
    case CUR_NONE:
        break;
    }

    V = v3d_add(V, v3d_scalar(v3d_cross(S, V), g->param.alpha));
    return v3d_scalar(V, 1.0 / (1.0 + g->param.alpha * g->param.alpha));
}

v3d step(size_t I, GLOBAL grid_t* g, v3d field, current_t cur, double dt, double norm_time) {
    #if defined(RK4)
    v3d rk1, rk2, rk3, rk4;
    rk1 = ds_dtau(I, g, field, v3d_s(0.0), cur, norm_time);
    rk2 = ds_dtau(I, g, field, v3d_scalar(rk1, dt / 2.0), cur, norm_time + dt / 2.0);
    rk3 = ds_dtau(I, g, field, v3d_scalar(rk2, dt / 2.0), cur, norm_time + dt / 2.0);
    rk4 = ds_dtau(I, g, field, v3d_scalar(rk3, dt), cur, norm_time + dt);
    return v3d_scalar(v3d_add(v3d_add(rk1, v3d_scalar(rk2, 2.0)), v3d_add(v3d_scalar(rk3, 2.0), rk4)), dt / 6.0);

    #elif defined(RK2)
    v3d rk1, rk2;
    rk1 = ds_dtau(I, g, field, v3d_s(0.0), cur, norm_time);
    rk2 = ds_dtau(I, g, field, v3d_scalar(rk1, dt), cur, norm_time + dt);
    return v3d_scalar(v3d_add(rk1, rk2), dt / 2.0);

    #elif defined(EULER)
    return v3d_scalar(ds_dtau(I, g, field, v3d_s(0.0), cur, norm_time), dt);

    #else
    return (v3d){0.0, 0.0, 0.0};
    #endif
}

double charge_old(size_t I, GLOBAL v3d* g, int rows, int cols, double dx, double dy, pbc_t pbc) {
    int col = I % cols;
    int row = (I - col) / cols;
    v3d R = get_pbc_v3d(row, col + 1, g, rows, cols, pbc),
        L = get_pbc_v3d(row, col - 1, g, rows, cols, pbc),
        U = get_pbc_v3d(row + 1, col, g, rows, cols, pbc),
        D = get_pbc_v3d(row - 1, col, g, rows, cols, pbc);
    
    v3d dgdx = v3d_scalar(v3d_sub(R, L), 0.5 / dx);
    v3d dgdy = v3d_scalar(v3d_sub(U, D), 0.5 / dy);
    return 1.0 / (4 * M_PI) * dx * dy * v3d_dot(v3d_cross(dgdx, dgdy), g[I]);
}

double Q_ijk(v3d mi, v3d mj, v3d mk) {
    double num = v3d_dot(mi, v3d_cross(mj, mk));
    double den = 1.0 + v3d_dot(mi, mj) + v3d_dot(mi, mk) + v3d_dot(mj, mk);
    return 2.0 * atan2(num, den);
}

double charge(size_t I, GLOBAL v3d* g, int rows, int cols, pbc_t pbc) {
     //https://iopscience.iop.org/article/10.1088/2633-1357/abad0c/pdf
    int col = I % cols;
    int row = (I - col) / cols;
    #if 0
    v3d m2 = get_pbc_v3d(row, col + 1, g, rows, cols, pbc),
        m4 = get_pbc_v3d(row + 1, col, g, rows, cols, pbc),
        m3 = get_pbc_v3d(row + 1, col + 1, g, rows, cols, pbc),
        m1 = get_pbc_v3d(row, col, g, rows, cols, pbc);
    /*double num1 = v3d_dot(m1, v3d_cross(m2, m4));
    double den1 = 1.0 + v3d_dot(m1, m2) + v3d_dot(m1, m4) + v3d_dot(m2, m4);
    double num2 = v3d_dot(m2, v3d_cross(m3, m4));
    double den2 = 1.0 + v3d_dot(m2, m3) + v3d_dot(m2, m4) + v3d_dot(m3, m4);

    double q_124 = 2.0 * atan2(num1, den1);
    double q_234 = 2.0 * atan2(num2, den2);*/
    double q_124 = Q_ijk(m1, m2, m4);
    double q_234 = Q_ijk(m2, m3, m4);
    

    return 1.0 / (4.0 * M_PI) * (q_124 + q_234);
    #else
    v3d m2 = get_pbc_v3d(row, col + 1, g, rows, cols, pbc),
        m3 = get_pbc_v3d(row + 1, col, g, rows, cols, pbc),
        m4 = get_pbc_v3d(row, col - 1, g, rows, cols, pbc),
        m5 = get_pbc_v3d(row - 1, col, g, rows, cols, pbc),
        m1 = get_pbc_v3d(row, col, g, rows, cols, pbc);

    /*double num1 = v3d_dot(m1, v3d_cross(m2, m3));
    double den1 = 1.0 + v3d_dot(m1, m2) + v3d_dot(m1, m3) + v3d_dot(m2, m3);
    double num2 = v3d_dot(m1, v3d_cross(m3, m4));
    double den2 = 1.0 + v3d_dot(m1, m3) + v3d_dot(m1, m4) + v3d_dot(m3, m4);
    double num3 = v3d_dot(m1, v3d_cross(m4, m5));
    double den3 = 1.0 + v3d_dot(m1, m4) + v3d_dot(m1, m5) + v3d_dot(m4, m5);
    double num4 = v3d_dot(m1, v3d_cross(m5, m2));
    double den4 = 1.0 + v3d_dot(m1, m5) + v3d_dot(m1, m2) + v3d_dot(m5, m2);

    double q_123 = 2.0 * atan2(num1, den1);
    double q_134 = 2.0 * atan2(num2, den2);
    double q_145 = 2.0 * atan2(num3, den3);
    double q_152 = 2.0 * atan2(num4, den4);*/

    double q_123 = Q_ijk(m1, m2, m3);
    double q_134 = Q_ijk(m1, m3, m4);
    double q_145 = Q_ijk(m1, m4, m5);
    double q_152 = Q_ijk(m1, m5, m2);

    return 1.0 / (8.0 * M_PI) * (q_123 + q_134 + q_145 + q_152);

    #endif
}


v3d B_emergent(size_t I, GLOBAL v3d* g, int rows, int cols, double dx, double dy, pbc_t pbc) {
    return v3d_c(0.0, 0.0, HBAR / QE * 4.0 * M_PI * charge_old(I, g, rows, cols, dx, dy, pbc) / (dx * dy));
}

v3d E_emergent(size_t I, GLOBAL v3d* current, GLOBAL v3d* before, GLOBAL v3d* after, int rows, int cols, double dx, double dy, double dt, pbc_t pbc) {
    v3d ret = v3d_s(0.0);
    int col = I % cols;
    int row = (I - col) / cols;
    v3d R = get_pbc_v3d(row, col + 1, current, rows, cols, pbc),
        L = get_pbc_v3d(row, col - 1, current, rows, cols, pbc),
        U = get_pbc_v3d(row + 1, col, current, rows, cols, pbc),
        D = get_pbc_v3d(row - 1, col, current, rows, cols, pbc),
        C = get_pbc_v3d(row, col, current, rows, cols, pbc);
    
    v3d dgdx = v3d_scalar(v3d_sub(R, L), 0.5 / dx);
    v3d dgdy = v3d_scalar(v3d_sub(U, D), 0.5 / dy);
    v3d dgdt = v3d_scalar(v3d_sub(after[I], before[I]), 0.5 / dt);
    ret.x = HBAR / QE * v3d_dot(C, v3d_cross(dgdx, dgdt));
    ret.y = HBAR / QE * v3d_dot(C, v3d_cross(dgdy, dgdt));
    return ret;
}

v3d velocity(size_t I, GLOBAL v3d* current, GLOBAL v3d* before, GLOBAL v3d* after, int rows, int cols, double dx, double dy, double dt, pbc_t pbc) {
    v3d Em = E_emergent(I, current, before, after, rows, cols, dx, dy, dt, pbc);
    v3d Bm = B_emergent(I, current, rows, cols, dx, dy, pbc);
    return (v3d){ Em.y / Bm.z, -Em.x / Bm.z, 0.0 };
}

v3d velocity_weighted(size_t I, GLOBAL v3d* current, GLOBAL v3d* before, GLOBAL v3d* after, int rows, int cols, double dx, double dy, double dt, pbc_t pbc) {
    v3d Em = E_emergent(I, current, before, after, rows, cols, dx, dy, dt, pbc);
    double factor = dx * dy * QE / (4.0 * M_PI * HBAR);
    return (v3d){ factor * Em.y, -factor * Em.x, 0.0 };
}

v3d gradient_descente_velocity(v3d g_p, v3d g_n, double dt) {
    return v3d_scalar(
		    v3d_sub(g_n, g_p),
		    1.0 / (2.0 * dt)
		    );
}

v3d gradient_descent_force(size_t I, GLOBAL grid_t *g_aux, v3d vel, GLOBAL v3d *g_c, v3d field, double J, double alpha, double beta) {
    v3d F = v3d_scalar(dH_dSi(I, g_c[I], g_c, &g_aux->param, g_aux->regions, g_aux->ani, field, 0), 1.0 / J);
    //v3d F = v3d_scalar(dH_dSi(I, v3d_s(0), g_aux, field, 0), 1.0 / J);
    F = v3d_sub(F, v3d_scalar(vel, alpha));
    F = v3d_sub(F, v3d_scalar(g_c[I], beta));
    return F;
}
#endif
