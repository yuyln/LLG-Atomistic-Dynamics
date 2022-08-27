#ifndef __FUNCS
#define __FUNCS

#include <grid.h>
#include <constants.h>

Vec PBCVec(int row, int col, const Vec* v, int rows, int cols, PBC pbc)
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

double HamiltonianI(size_t I, Grid *g, Vec field)
{
    int col = I % g->param.cols;
    int row = (I - col) / g->param.cols;
    Vec C = PBCVec(row, col, g->grid, g->param.rows, g->param.cols, g->param.pbc),
        R = PBCVec(row, col + 1, g->grid, g->param.rows, g->param.cols, g->param.pbc),
        L = PBCVec(row, col - 1, g->grid, g->param.rows, g->param.cols, g->param.pbc),
        U = PBCVec(row + 1, col, g->grid, g->param.rows, g->param.cols, g->param.pbc),
        D = PBCVec(row - 1, col, g->grid, g->param.rows, g->param.cols, g->param.pbc);

    Vec DMR = DMVec(0, 1, g->param.dm_type, g->param.dm),
        DML = DMVec(0, -1, g->param.dm_type, g->param.dm),
        DMU = DMVec(1, 0, g->param.dm_type, g->param.dm),
        DMD = DMVec(-1, 0, g->param.dm_type, g->param.dm);

    double out = -g->param.mu_s * VecDot(C, field);

    out += -0.5 * g->param.exchange * (VecDot(C, R)+
                                       VecDot(C, L)+
                                       VecDot(C, U)+
                                       VecDot(C, D));

    out += -0.5 * (VecDot(DMR, VecCross(C, R))+
                   VecDot(DML, VecCross(C, L))+
                   VecDot(DMU, VecCross(C, U))+
                   VecDot(DMD, VecCross(C, D)));

    out += -g->ani[I].K_1 * (VecDot(C, g->ani[I].dir)) * (VecDot(C, g->ani[I].dir));

    out += -g->param.cubic_ani * (C.x * C.x * C.x * C.x+
                                  C.y * C.y * C.y * C.y+
                                  C.z * C.z * C.z * C.z);

    return out;
}

/*
double Hamiltoninij(int row, int col, Grid *g, Vec field)
{
    size_t I = (size_t) (row * g->param.cols + col);
    Vec C = PBCVec(row, col, g->grid, g->param.rows, g->param.cols, g->param.pbc),
        R = PBCVec(row, col + 1, g->grid, g->param.rows, g->param.cols, g->param.pbc),
        L = PBCVec(row, col - 1, g->grid, g->param.rows, g->param.cols, g->param.pbc),
        U = PBCVec(row + 1, col, g->grid, g->param.rows, g->param.cols, g->param.pbc),
        D = PBCVec(row - 1, col, g->grid, g->param.rows, g->param.cols, g->param.pbc);

    Vec DMR = DMVec(0, 1, g->param.dm_type, g->param.dm),
        DML = DMVec(0, -1, g->param.dm_type, g->param.dm),
        DMU = DMVec(1, 0, g->param.dm_type, g->param.dm),
        DMD = DMVec(-1, 0, g->param.dm_type, g->param.dm);

    double out = -g->param.mu_s * VecDot(C, field);

    out += -0.5 * g->param.exchange * (VecDot(C, R)+
                                       VecDot(C, L)+
                                       VecDot(C, U)+
                                       VecDot(C, D));

    out += -0.5 * (VecDot(DMR, VecCross(C, R))+
                   VecDot(DML, VecCross(C, L))+
                   VecDot(DMU, VecCross(C, U))+
                   VecDot(DMD, VecCross(C, D)));

    out += -g->ani[I].K_1 * VecDot(C, g->ani[I].dir) * VecDot(C, g->ani[I].dir);

    out += -g->param.cubic_ani * (C.x * C.x * C.x * C.x+
                                  C.y * C.y * C.y * C.y+
                                  C.z * C.z * C.z * C.z);

    return out;
}
*/

double Hamiltonian(Grid *g, Vec field)
{
    double ret = 0.0;
    for (size_t I = 0; I < g->param.total; ++I)
        ret += HamiltonianI(I, g, field);
    return ret;
}

void GridNormalizeI(size_t I, Grid *g)
{
    if (g->pinning[I].fixed)
        g->grid[I] = g->pinning[I].dir;
    else
        g->grid[I] = VecNormalize(g->grid[I]);
}

Vec VecDotGradVecI(size_t I, Vec v, Vec *g, int rows, int cols, double dx, double dy, PBC pbc)
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

Vec dHdSI(size_t I, Vec C, Grid *g, Vec field)
{
    int col = I % g->param.cols;
    int row = (I - col) / g->param.cols;
    Vec ret = VecFromScalar(0.0);

    Vec R = PBCVec(row, col + 1, g->grid, g->param.rows, g->param.cols, g->param.pbc),
        L = PBCVec(row, col - 1, g->grid, g->param.rows, g->param.cols, g->param.pbc),
        U = PBCVec(row + 1, col, g->grid, g->param.rows, g->param.cols, g->param.pbc),
        D = PBCVec(row - 1, col, g->grid, g->param.rows, g->param.cols, g->param.pbc);
    
    Vec DMR = DMVec(0, 1, g->param.dm_type, -g->param.dm), //put (-) here to remove from later
        DML = DMVec(0, -1, g->param.dm_type, -g->param.dm),
        DMU = DMVec(1, 0, g->param.dm_type, -g->param.dm),
        DMD = DMVec(-1, 0, g->param.dm_type, -g->param.dm);
    
    Vec exchange = VecScalar(R, -g->param.exchange);
    exchange = VecAdd(exchange, VecScalar(L, -g->param.exchange));
    exchange = VecAdd(exchange, VecScalar(U, -g->param.exchange));
    exchange = VecAdd(exchange, VecScalar(D, -g->param.exchange));
    ret = VecAdd(ret, exchange);

    Vec dm = VecCross(R, DMR);
    dm = VecAdd(dm, VecCross(L, DML));
    dm = VecAdd(dm, VecCross(U, DMU));
    dm = VecAdd(dm, VecCross(D, DMD));
    ret = VecAdd(ret, dm);

    Vec ani = VecScalar(g->ani[I].dir, -2.0 * g->ani[I].K_1 * VecDot(C, g->ani[I].dir));
    ret = VecAdd(ret, ani);
    Vec cub_ani = VecFrom(-4.0 * g->param.cubic_ani * C.x * C.x * C.x,
                          -4.0 * g->param.cubic_ani * C.y * C.y * C.y,
                          -4.0 * g->param.cubic_ani * C.z * C.z * C.z);
    ret = VecAdd(ret, cub_ani);
    
    ret = VecSub(ret, VecScalar(field, g->param.mu_s));

    return ret;
}

Vec dSdTauI(size_t I, Grid *g, Vec field, Vec dS, Current cur)
{
    Vec S = VecAdd(g->grid[I], dS);
    Vec Heff = VecScalar(dHdSI(I, S, g, field), -1.0 / g->param.mu_s);
    double J_abs = g->param.exchange * (g->param.exchange < 0? -1.0: 1.0);

    Vec V = VecScalar(VecCross(S, Heff), -g->param.gamma * HBAR / J_abs);

    switch (cur.type)
    {
    case CUR_CPP:
    {
        Vec cur_local = VecScalar(VecCross(S, cur.j), g->param.gamma * HBAR * cur.p * g->param.lattice * g->param.avg_spin / (cur.thick * g->param.mu_s));
        V = VecAdd(V, VecCross(S, cur_local));
        V = VecAdd(V, cur_local);
        break;
    }
    case CUR_STT:
    {
        Vec cur_local = VecDotGradVecI(I, cur.j, g->grid, g->param.rows, g->param.cols, g->param.lattice, g->param.lattice, g->param.pbc);
        V = VecAdd(V, VecScalar(cur_local, cur.p * g->param.lattice));
        V = VecAdd(V, VecScalar(VecCross(S, cur_local), cur.p * cur.beta * g->param.lattice / g->param.avg_spin));
        break;
    }
    case CUR_BOTH:
    {
        Vec cur_local = VecScalar(VecCross(S, cur.j), g->param.gamma * HBAR * cur.p * g->param.lattice * g->param.avg_spin / (cur.thick * g->param.mu_s));
        V = VecAdd(V, VecCross(S, cur_local));
        V = VecAdd(V, cur_local);
        cur_local = VecDotGradVecI(I, cur.j, g->grid, g->param.rows, g->param.cols, g->param.lattice, g->param.lattice, g->param.pbc);
        V = VecAdd(V, VecScalar(cur_local, cur.p * g->param.lattice));
        V = VecAdd(V, VecScalar(VecCross(S, cur_local), cur.p * cur.beta * g->param.lattice / g->param.avg_spin));
        break;
    }
    case CUR_NONE:
        break;
    }

    V = VecAdd(V, VecScalar(VecCross(S, V), g->param.alpha));
    return VecScalar(V, 1.0 / (1.0 + g->param.alpha * g->param.alpha));
}

Vec StepI(size_t I, Grid *g, Vec field, Current cur, double dt)
{
    #if defined(RK4)
    Vec rk1, rk2, rk3, rk4;
    rk1 = dSdTauI(I, g, field, VecFromScalar(0.0), cur);
    rk2 = dSdTauI(I, g, field, VecScalar(rk1, dt / 2.0), cur);
    rk3 = dSdTauI(I, g, field, VecScalar(rk2, dt / 2.0), cur);
    rk4 = dSdTauI(I, g, field, VecScalar(rk3, dt), cur);
    return VecScalar(VecAdd(VecAdd(rk1, VecScalar(rk2, 2.0)), VecAdd(VecScalar(rk3, 2.0), rk4)), dt / 6.0);

    #elif defined(RK2)
    Vec rk1, rk2;
    rk1 = dSdTauI(I, g, field, VecFromScalar(0.0), cur);
    rk2 = dSdTauI(I, g, field, VecScalar(rk1, dt), cur);
    return VecScalar(VecAdd(rk1, rk2), dt / 2.0);

    #elif defined(EULER)
    return VecScalar(dSdTauI(I, g, field, VecFromScalar(0.0), cur), dt);

    #else
    return (Vec){0.0, 0.0, 0.0};
    #endif
}

double ChargeI(size_t I, Vec *g, int rows, int cols, double dx, double dy, PBC pbc)
{
    int col = I % cols;
    int row = (I - col) / cols;
    Vec R = PBCVec(row, col + 1, g, rows, cols, pbc),
        L = PBCVec(row, col - 1, g, rows, cols, pbc),
        U = PBCVec(row + 1, col, g, rows, cols, pbc),
        D = PBCVec(row - 1, col, g, rows, cols, pbc);
    
    Vec dgdx = VecScalar(VecSub(R, L), 1.0 / dx);
    Vec dgdy = VecScalar(VecSub(U, D), 1.0 / dy);
    return 1.0 / (4 * M_PI) * dx * dy * VecDot(VecCross(dgdx, dgdy), g[I]);
}
#endif