#ifndef __FUNCS
#define __FUNCS

#include <grid.h>
#include <constants.h>

inline Current GenCurI(size_t I, GLOBAL GridParam* g, Current base, double norm_time)
{
    UNUSED(I); UNUSED(g); UNUSED(norm_time);
    return base;
}

inline Vec GenFieldI(size_t I, GLOBAL GridParam* g, Vec base, double norm_time)
{
    UNUSED(I); UNUSED(g); UNUSED(norm_time);
    return base;
}

Vec LinearInterp(Vec v1, Vec v2, double t)
{
    return VecAdd(v1, VecScalar(VecSub(v2, v1), t));
}

Vec BilinearInterp(Vec v00, Vec v10, Vec v11, Vec v01, double u, double v)
{
    Vec b = VecSub(v10, v00);
    Vec c = VecSub(v01, v00);
    Vec d = VecSub(v11, v10);
    d = VecSub(d, v01);
    d = VecAdd(d, v00);

    Vec ret = v00;
    ret = VecAdd(ret, VecScalar(b, u));
    ret = VecAdd(ret, VecScalar(c, v));
    ret = VecAdd(ret, VecScalar(d, u * v));
    return ret;
}

Vec PBCVec(int row, int col, const GLOBAL Vec* v, int rows, int cols, PBC pbc)
{
    switch (pbc.pbc_type)
    {
    case PBC_NONE:
        if (row >= rows || row < 0 || col >= cols || col < 0)
            return pbc.dir;
        break;
    
    case PBC_X:
        if (row >= rows || row < 0)
            return pbc.dir;
        if (col >= cols)
            col = col % cols;
        else if (col < 0)
            col = (col * (1 - cols)) % cols;
        break;

    case PBC_Y:
        if (col >= cols || col < 0)
            return pbc.dir;
        
        if (row >= rows)
            row = row % rows;
        else if (row < 0)
            row = (row * (1 - rows)) % rows;
        break;
    
    case PBC_XY:
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

Vec DMVec(int drow, int dcol, DM_TYPE dm_type, double dm)
{
    switch (dm_type)
    {
    case R_ij:
        if (drow * drow + dcol * dcol > 1)
            return VecNormalizeTo(VecFrom(dcol, drow, 0), dm);
        return VecFrom(dcol * dm, drow * dm, 0.0);

    case Z_CROSS_R_ij:
        if (drow * drow + dcol * dcol > 1)
            return VecNormalizeTo(VecFrom(-drow, dcol, 0), dm);
        return VecFrom(-drow * dm, dcol * dm, 0.0);
    }
    return VecFromScalar(0.0);
}

//double HamiltonianI(size_t I, GLOBAL Grid* g, Vec field)
double HamiltonianI(size_t I, GLOBAL Vec *v, GLOBAL GridParam *param, GLOBAL Anisotropy *anis, GLOBAL RegionParam *regions, Vec field)
{
    int col = I % param->cols;
    int row = (I - col) / param->cols;
    Vec C = PBCVec(row, col, v, param->rows, param->cols, param->pbc),
        R = PBCVec(row, col + 1, v, param->rows, param->cols, param->pbc),
        L = PBCVec(row, col - 1, v, param->rows, param->cols, param->pbc),
        U = PBCVec(row + 1, col, v, param->rows, param->cols, param->pbc),
        D = PBCVec(row - 1, col, v, param->rows, param->cols, param->pbc);

    Vec DMR = DMVec(0, 1, regions[I].dm_type, param->dm * regions[I].dm_mult),
        DML = DMVec(0, -1, regions[I].dm_type, param->dm * regions[I].dm_mult),
        DMU = DMVec(1, 0, regions[I].dm_type, param->dm * regions[I].dm_mult),
        DMD = DMVec(-1, 0, regions[I].dm_type, param->dm * regions[I].dm_mult);

    double out = -param->mu_s * VecDot(C, GenFieldI(I, param, field, 0.0)) * regions[I].field_mult;


    out += -0.5 * param->exchange * regions[I].exchange_mult * (VecDot(C, R)+
                                       VecDot(C, L)+
                                       VecDot(C, U)+
                                       VecDot(C, D));

    out += -0.5 * (VecDot(DMR, VecCross(C, R))+
                   VecDot(DML, VecCross(C, L))+
                   VecDot(DMU, VecCross(C, U))+
                   VecDot(DMD, VecCross(C, D)));

    out += -anis[I].K_1 * (VecDot(C, anis[I].dir)) * (VecDot(C, anis[I].dir));

    out += -param->cubic_ani * (C.x * C.x * C.x * C.x+
                                  C.y * C.y * C.y * C.y+
                                  C.z * C.z * C.z * C.z);

    return out;
}

double Hamiltonian(GLOBAL Grid* g, Vec field)
{
    double ret = 0.0;
    for (size_t I = 0; I < g->param.total; ++I)
        ret += HamiltonianI(I, g->grid, &g->param, g->ani, g->regions, field);
    return ret;
}

void GridNormalizeI(size_t I, GLOBAL Vec* v, GLOBAL Pinning *pin)
{
    if (pin[I].fixed)
        v[I] = pin[I].dir;
    else
        v[I] = VecNormalize(v[I]);
}

Vec VecDotGradVecI(size_t I, Vec v, GLOBAL Vec* g, int rows, int cols, double dx, double dy, PBC pbc)
{
    Vec ret;
    int col = I % cols;
    int row = (I - col) / cols;
    Vec R = PBCVec(row, col + 1, g, rows, cols, pbc),
        L = PBCVec(row, col - 1, g, rows, cols, pbc),
        U = PBCVec(row + 1, col, g, rows, cols, pbc),
        D = PBCVec(row - 1, col, g, rows, cols, pbc);
    
    ret.x = v.x * (R.x - L.x) / (2.0 * dx)+
            v.y * (U.x - D.x) / (2.0 * dy);

    ret.y = v.x * (R.y - L.y) / (2.0 * dx)+
            v.y * (U.y - D.y) / (2.0 * dy);
    
    ret.z = v.x * (R.z - L.z) / (2.0 * dx)+
            v.y * (U.z - D.z) / (2.0 * dy);
    
    return ret;
}

//Vec dHdSI(size_t I, Vec C, GLOBAL Grid* g, Vec field, double norm_time)
Vec dHdSI(size_t I, Vec C, GLOBAL Vec *v, GLOBAL GridParam *param, GLOBAL RegionParam *regions, GLOBAL Anisotropy *anis, Vec field, double norm_time)
{
    int col = I % param->cols;
    int row = (I - col) / param->cols;
    Vec ret = VecFromScalar(0.0);

    Vec R = PBCVec(row, col + 1, v, param->rows, param->cols, param->pbc),
        L = PBCVec(row, col - 1, v, param->rows, param->cols, param->pbc),
        U = PBCVec(row + 1, col, v, param->rows, param->cols, param->pbc),
        D = PBCVec(row - 1, col, v, param->rows, param->cols, param->pbc);
    
    Vec DMR = DMVec(0, 1, regions[I].dm_type, -param->dm * regions[I].dm_mult), //put (-) here to remove from later
        DML = DMVec(0, -1, regions[I].dm_type, -param->dm * regions[I].dm_mult),
        DMU = DMVec(1, 0, regions[I].dm_type, -param->dm * regions[I].dm_mult),
        DMD = DMVec(-1, 0, regions[I].dm_type, -param->dm * regions[I].dm_mult);
    
    Vec exchange = VecScalar(R, -param->exchange * regions[I].exchange_mult);
    exchange = VecAdd(exchange, VecScalar(L, -param->exchange * regions[I].exchange_mult));
    exchange = VecAdd(exchange, VecScalar(U, -param->exchange * regions[I].exchange_mult));
    exchange = VecAdd(exchange, VecScalar(D, -param->exchange * regions[I].exchange_mult));
    ret = VecAdd(ret, exchange);

    Vec dm = VecCross(R, DMR);
    dm = VecAdd(dm, VecCross(L, DML));
    dm = VecAdd(dm, VecCross(U, DMU));
    dm = VecAdd(dm, VecCross(D, DMD));
    ret = VecAdd(ret, dm);

    Vec ani = VecScalar(anis[I].dir, -2.0 * anis[I].K_1 * VecDot(C, anis[I].dir));
    ret = VecAdd(ret, ani);
    Vec cub_ani = VecFrom(-4.0 * param->cubic_ani * C.x * C.x * C.x,
                          -4.0 * param->cubic_ani * C.y * C.y * C.y,
                          -4.0 * param->cubic_ani * C.z * C.z * C.z);
    ret = VecAdd(ret, cub_ani);
    
    ret = VecSub(ret, VecScalar(GenFieldI(I, param, field, norm_time), param->mu_s * regions[I].field_mult));

    return ret;
}

Vec dSdTauI(size_t I, GLOBAL Grid* g, Vec field, Vec dS, Current cur, double norm_time)
{
    Vec S = VecAdd(g->grid[I], dS);
    Vec Heff = VecScalar(dHdSI(I, S, g->grid, &g->param, g->regions, g->ani, field, norm_time), -1.0 / g->param.mu_s);
    //Vec Heff = VecScalar(dHdSI(I, S, g, field, norm_time), -1.0 / g->param.mu_s);
    double J_abs = g->param.exchange * g->regions[I].exchange_mult * (g->param.exchange * g->regions[I].exchange_mult < 0? -1.0: 1.0);

    Vec V = VecScalar(VecCross(S, Heff), -g->param.gamma * HBAR / J_abs);

    switch (cur.type)
    {
    case CUR_CPP:
    {
        cur = GenCurI(I, &g->param, cur, norm_time);
        double factor = g->param.gamma * HBAR * cur.p * g->param.lattice * g->param.avg_spin / (cur.thick * g->param.mu_s);
        Vec cur_local = VecScalar(VecCross(cur.j, S), factor);
        V = VecAdd(V, VecCross(S, cur_local));
        V = VecAdd(V, VecScalar(cur_local, cur.beta));
        break;
    }
    case CUR_STT:
    {
        cur = GenCurI(I, &g->param, cur, norm_time);
        Vec cur_local = VecDotGradVecI(I, cur.j, g->grid, g->param.rows, g->param.cols, g->param.lattice, g->param.lattice, g->param.pbc);
        V = VecAdd(V, VecScalar(cur_local, cur.p * g->param.lattice));
        V = VecSub(V, VecScalar(VecCross(S, cur_local), cur.p * cur.beta * g->param.lattice / g->param.avg_spin));
        break;
    }
    case CUR_BOTH:
    {
        cur = GenCurI(I, &g->param, cur, norm_time);
        double factor = g->param.gamma * HBAR * cur.p * g->param.lattice * g->param.avg_spin / (cur.thick * g->param.mu_s);
        Vec cur_local = VecScalar(VecCross(cur.j, S), factor);
        V = VecAdd(V, VecCross(S, cur_local));
        V = VecAdd(V, VecScalar(cur_local, cur.beta));

        cur_local = VecDotGradVecI(I, cur.j, g->grid, g->param.rows, g->param.cols, g->param.lattice, g->param.lattice, g->param.pbc);
        V = VecAdd(V, VecScalar(cur_local, cur.p * g->param.lattice));
        V = VecSub(V, VecScalar(VecCross(S, cur_local), cur.p * cur.beta * g->param.lattice / g->param.avg_spin));
        break;
    }
    case CUR_NONE:
        break;
    }

    V = VecAdd(V, VecScalar(VecCross(S, V), g->param.alpha));
    return VecScalar(V, 1.0 / (1.0 + g->param.alpha * g->param.alpha));
}

Vec StepI(size_t I, GLOBAL Grid* g, Vec field, Current cur, double dt, double norm_time)
{
    #if defined(RK4)
    Vec rk1, rk2, rk3, rk4;
    rk1 = dSdTauI(I, g, field, VecFromScalar(0.0), cur, norm_time);
    rk2 = dSdTauI(I, g, field, VecScalar(rk1, dt / 2.0), cur, norm_time + dt / 2.0);
    rk3 = dSdTauI(I, g, field, VecScalar(rk2, dt / 2.0), cur, norm_time + dt / 2.0);
    rk4 = dSdTauI(I, g, field, VecScalar(rk3, dt), cur, norm_time + dt);
    return VecScalar(VecAdd(VecAdd(rk1, VecScalar(rk2, 2.0)), VecAdd(VecScalar(rk3, 2.0), rk4)), dt / 6.0);

    #elif defined(RK2)
    Vec rk1, rk2;
    rk1 = dSdTauI(I, g, field, VecFromScalar(0.0), cur, norm_time);
    rk2 = dSdTauI(I, g, field, VecScalar(rk1, dt), cur, norm_time + dt);
    return VecScalar(VecAdd(rk1, rk2), dt / 2.0);

    #elif defined(EULER)
    return VecScalar(dSdTauI(I, g, field, VecFromScalar(0.0), cur, norm_time), dt);

    #else
    return (Vec){0.0, 0.0, 0.0};
    #endif
}

double ChargeI_old(size_t I, GLOBAL Vec* g, int rows, int cols, double dx, double dy, PBC pbc)
{
    int col = I % cols;
    int row = (I - col) / cols;
    Vec R = PBCVec(row, col + 1, g, rows, cols, pbc),
        L = PBCVec(row, col - 1, g, rows, cols, pbc),
        U = PBCVec(row + 1, col, g, rows, cols, pbc),
        D = PBCVec(row - 1, col, g, rows, cols, pbc);
    
    Vec dgdx = VecScalar(VecSub(R, L), 0.5 / dx);
    Vec dgdy = VecScalar(VecSub(U, D), 0.5 / dy);
    return 1.0 / (4 * M_PI) * dx * dy * VecDot(VecCross(dgdx, dgdy), g[I]);
}


double ChargeI(size_t I, GLOBAL Vec* g, int rows, int cols, PBC pbc)
{
     //https://iopscience.iop.org/article/10.1088/2633-1357/abad0c/pdf
    int col = I % cols;
    int row = (I - col) / cols;
    Vec m2 = PBCVec(row, col + 1, g, rows, cols, pbc),
        m4 = PBCVec(row + 1, col, g, rows, cols, pbc),
        m3 = PBCVec(row + 1, col + 1, g, rows, cols, pbc),
        m1 = PBCVec(row, col, g, rows, cols, pbc);
    double num1 = VecDot(m1, VecCross(m2, m4));
    double den1 = 1.0 + VecDot(m1, m2) + VecDot(m1, m4) + VecDot(m2, m4);
    double num2 = VecDot(m2, VecCross(m3, m4));
    double den2 = 1.0 + VecDot(m2, m3) + VecDot(m2, m4) + VecDot(m3, m4);

    double q_124 = 2.0 * atan2(num1, den1);
    double q_234 = 2.0 * atan2(num2, den2);

    return 1.0 / (4.0 * M_PI) * (q_124 + q_234);
}


Vec BemI(size_t I, GLOBAL Vec* g, int rows, int cols, double dx, double dy, PBC pbc)
{
    return VecFrom(0.0, 0.0, HBAR / QE * 4.0 * M_PI * ChargeI_old(I, g, rows, cols, dx, dy, pbc) / (dx * dy));
}

Vec EemI(size_t I, GLOBAL Vec* current, GLOBAL Vec* before, GLOBAL Vec* after, int rows, int cols, double dx, double dy, double dt, PBC pbc)
{
    Vec ret = VecFromScalar(0.0);
    int col = I % cols;
    int row = (I - col) / cols;
    Vec R = PBCVec(row, col + 1, current, rows, cols, pbc),
        L = PBCVec(row, col - 1, current, rows, cols, pbc),
        U = PBCVec(row + 1, col, current, rows, cols, pbc),
        D = PBCVec(row - 1, col, current, rows, cols, pbc),
        C = PBCVec(row, col, current, rows, cols, pbc);
    
    Vec dgdx = VecScalar(VecSub(R, L), 0.5 / dx);
    Vec dgdy = VecScalar(VecSub(U, D), 0.5 / dy);
    Vec dgdt = VecScalar(VecSub(after[I], before[I]), 0.5 / dt);
    ret.x = HBAR / QE * VecDot(C, VecCross(dgdx, dgdt));
    ret.y = HBAR / QE * VecDot(C, VecCross(dgdy, dgdt));
    return ret;
}

Vec VelI(size_t I, GLOBAL Vec* current, GLOBAL Vec* before, GLOBAL Vec* after, int rows, int cols, double dx, double dy, double dt, PBC pbc)
{
    Vec Em = EemI(I, current, before, after, rows, cols, dx, dy, dt, pbc);
    Vec Bm = BemI(I, current, rows, cols, dx, dy, pbc);
    return (Vec){ Em.y / Bm.z, -Em.x / Bm.z, 0.0 };
}

Vec VelWeightedI(size_t I, GLOBAL Vec* current, GLOBAL Vec* before, GLOBAL Vec* after, int rows, int cols, double dx, double dy, double dt, PBC pbc)
{
    Vec Em = EemI(I, current, before, after, rows, cols, dx, dy, dt, pbc);
    double factor = dx * dy * QE / (4.0 * M_PI * HBAR);
    return (Vec){ factor * Em.y, -factor * Em.x, 0.0 };
}

Vec GradientDescentVelocity(Vec g_p, Vec g_n, double dt)
{
    return VecScalar(
		    VecSub(g_n, g_p),
		    1.0 / (2.0 * dt)
		    );
}

Vec GradientDescentForce(size_t I, GLOBAL Grid *g_aux, Vec vel, GLOBAL Vec *g_c, Vec field, double J, double alpha, double beta)
{
    Vec F = VecScalar(dHdSI(I, g_c[I], g_c, &g_aux->param, g_aux->regions, g_aux->ani, field, 0), 1.0 / J);
    //Vec F = VecScalar(dHdSI(I, VecFromScalar(0), g_aux, field, 0), 1.0 / J);
    F = VecSub(F, VecScalar(vel, alpha));
    F = VecSub(F, VecScalar(g_c[I], beta));
    return F;
}
#endif
