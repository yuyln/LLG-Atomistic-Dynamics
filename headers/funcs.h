#ifndef __FUNCS
#define __FUNCS

#include <grid.h>

Vec PBCVec(int row, int col, const Vec* v, int rows, int cols, PBC pbc)
{
    switch (pbc.pbc_type)
    {
    case NONE:
        if (row >= rows || row < 0 || col >= cols || col < 0)
            return pbc.dir;
        break;
    
    case X:
        if (row >= rows || row < 0)
            return pbc.dir;
        if (col >= cols)
            col = col % cols;
        else if (col < 0)
            col = (col * (1 - cols)) % cols;
        break;

    case Y:
        if (col >= cols || col < 0)
            return pbc.dir;
        
        if (row >= rows)
            row = row % rows;
        else if (row < 0)
            row = (row * (1 - rows)) % rows;
        break;
    
    case XY:
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
#endif