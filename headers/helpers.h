#ifndef __HELP
#define __HELP

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>

#include <vec.h>
#include <constants.h>
#include <grid.h>
#include <parserL.h>
#include <funcs.h>
#include <time.h>
#include <opencl_kernel.h>
#define OPENCLWRAPPER_IMPLEMENTATION
#include <opencl_wrapper.h>
#define PROFILER_IMPLEMENTATION
#include <profiler.h>

double myrandom()
{
    return (double)rand() / (double)RAND_MAX;
}

double random_range(double min, double max)
{
    return myrandom() * (max - min) + min;
}

void FreeGrid(Grid *g)
{
    if (g->grid)
        free(g->grid);

    if (g->ani)
        free(g->ani);
    
    if (g->pinning)
        free(g->pinning);
}

void CopyGrid(Grid *to, Grid *from)
{
    FreeGrid(to);
    memcpy(&to->param, &from->param, sizeof(GridParam));

    to->grid = (Vec*)calloc(from->param.total, sizeof(Vec));
    to->ani = (Anisotropy*)calloc(from->param.total, sizeof(Anisotropy));
    to->pinning = (Pinning*)calloc(from->param.total, sizeof(Pinning));

    memcpy(to->grid, from->grid, from->param.total * sizeof(Vec));
    memcpy(to->ani, from->ani, from->param.total * sizeof(Anisotropy));
    memcpy(to->pinning, from->pinning, from->param.total * sizeof(Pinning));
}

void CopyToExistingGrid(Grid *to, Grid *from)
{
    memcpy(to->grid, from->grid, from->param.total * sizeof(Vec));
    memcpy(to->ani, from->ani, from->param.total * sizeof(Anisotropy));
    memcpy(to->pinning, from->pinning, from->param.total * sizeof(Pinning));
}

void CopySpinsToExistingGrid(Grid *to, Grid *from)
{
    memcpy(to->grid, from->grid, from->param.total * sizeof(Vec));
}

Grid InitNullGrid()
{
    Grid ret;
    memset(&ret, 0, sizeof(Grid));
    return ret;
}

int FindRowsFile(const char *path)
{
    FILE* f = fopen(path, "rb");

    if (!f)
    {
        fprintf(stderr, "Could not open file %s: %s\n", path, strerror(errno));
        exit(1);
    }

    fseek(f, 0, SEEK_SET);
    fseek(f, 0, SEEK_END);

    size_t file_size = ftell(f);
    char* file_data = (char*)malloc(file_size + 1);

    fseek(f, 0, SEEK_SET);
    fread(file_data, 1, file_size, f);

    file_data[file_size] = '\0';

    char* ptr = file_data;
    int rows = 0;

    while(*ptr)
        if (*ptr++ == '\n')
            ++rows;
    
    if (*--ptr != '\n')
        rows++;

    free(file_data);
    fclose(f);
    return rows;
}

Vec* InitVecGridFromFile(const char* path, int *rows, int *cols)
{
    int rows_ = FindRowsFile(path);
    *rows = rows_;
    StartParse(path);

    int cols_ = parser_global_n / (3 * rows_);
    *cols = cols_;
    Vec* ret = (Vec*)calloc(rows_ * cols_, sizeof(Vec));
    for (size_t I = 0; I < parser_global_n; I += 3)
    {
        int j = (I / 3) % cols_;
        int i = rows_ - 1 - (I / 3 - j) / cols_;
        ret[i * cols_ + j] = VecNormalize(VecFrom(strtod(parser_global_state[I], NULL),
                                                  strtod(parser_global_state[I + 1], NULL),
                                                  strtod(parser_global_state[I + 2], NULL)));
    }
    EndParse();
    return ret;
}

Vec* InitVecGridRandom(size_t rows, size_t cols)
{
    Vec* ret = (Vec*)calloc(rows * cols, sizeof(Vec));
    for (size_t I = 0; I < rows * cols; ++I)
        ret[I] = VecNormalize(VecFrom(random_range(-1.0, 1.0), 
                                      random_range(-1.0, 1.0), 
                                      random_range(-1.0, 1.0)));
    return ret;
}

Grid InitGridFromFile(const char* path)
{
    Grid out = InitNullGrid();
    out.grid = InitVecGridFromFile(path, &out.param.rows, &out.param.cols);
    out.param.total = out.param.rows * out.param.cols;

    out.ani = (Anisotropy*)calloc(out.param.total, sizeof(Anisotropy));
    out.pinning = (Pinning*)calloc(out.param.total, sizeof(Pinning));
    return out;
}

Grid InitGridRandom(int rows, int cols)
{
    Grid ret = InitNullGrid();
    ret.param.rows = rows;
    ret.param.cols = cols;
    ret.param.total = rows * cols;
    ret.grid = InitVecGridRandom(rows, cols);
    ret.ani = (Anisotropy*)calloc(ret.param.total, sizeof(Anisotropy));
    ret.pinning = (Pinning*)calloc(ret.param.total, sizeof(Pinning));
    return ret;
}

size_t FindGridSize(const Grid* g)
{
    size_t param = sizeof(GridParam);
    size_t grid_vec = g->param.total * sizeof(Vec);
    size_t grid_pinning = g->param.total * sizeof(Pinning);
    size_t grid_ani = g->param.total * sizeof(Anisotropy);
    return param + grid_vec + grid_pinning + grid_ani;
}

void PrintVecGridToFile(const char* path, Vec* v, int rows, int cols)
{
    FILE *f = fopen(path, "wb");

    if (!f)
    {
        fprintf(stderr, "Could not open file %s: %s\n", path, strerror(errno));
        exit(1);
    }

    for (int row = rows - 1; row >= 1; --row)
    {
        for (int col = 0; col < cols - 1; ++col)
        {
            fprintf(f, "%.15f\t%.15f\t%.15f\t", v[row * cols + col].x, v[row * cols + col].y, v[row * cols + col].z);
        }
        int col = cols - 1;
        fprintf(f, "%.15f\t%.15f\t%.15f\n", v[row * cols + col].x, v[row * cols + col].y, v[row * cols + col].z);
    }

    int row = 0;

    for (int col = 0; col < cols - 1; ++col)
    {
        fprintf(f, "%.15f\t%.15f\t%.15f\t", v[row * cols + col].x, v[row * cols + col].y, v[row * cols + col].z);
    }
    int col = cols - 1;
    fprintf(f, "%.15f\t%.15f\t%.15f", v[row * cols + col].x, v[row * cols + col].y, v[row * cols + col].z);
    fclose(f);
}

void GetGridParam(const char* path, GridParam* g)
{
    StartParse(path);

    g->exchange = GetValueDouble("EXCHANGE");
    g->dm = GetValueDouble("DMI");
    g->lattice = GetValueDouble("LATTICE");
    g->cubic_ani = GetValueDouble("CUBIC");
    g->lande = GetValueDouble("LANDE");
    g->avg_spin = GetValueDouble("SPIN");
    g->mu_s = g->lande * MU_B * g->avg_spin;
    g->dm_type = GetValueInt("DM_TYPE", 10);
    g->pbc.pbc_type = GetValueInt("PBC_TYPE", 10);
    g->pbc.dir.x = GetValueDouble("PBC_X");
    g->pbc.dir.y = GetValueDouble("PBC_Y");
    g->pbc.dir.z = GetValueDouble("PBC_Z");

    EndParse();
}

Vec FieldJouleToTesla(Vec field, double mu_s)
{
    return VecScalar(field, 1.0 / mu_s);
}

Vec FieldTeslaToJoule(Vec field, double mu_s)
{
    return VecScalar(field, mu_s);
}

void WriteFullGridBuffer(cl_command_queue q, cl_mem buffer, Grid *g)
{
    size_t off = 0;
    WriteBuffer(buffer, &g->param, sizeof(GridParam), off, q);
    off += sizeof(GridParam);
    WriteBuffer(buffer, g->grid, sizeof(Vec) * g->param.total, off, q);
    off += sizeof(Vec) * g->param.total;
    WriteBuffer(buffer, g->ani, sizeof(Anisotropy) * g->param.total, off, q);
    off += sizeof(Anisotropy) * g->param.total;
    WriteBuffer(buffer, g->pinning, sizeof(Pinning) * g->param.total, off, q);
    off += sizeof(Pinning) * g->param.total;
}

void WriteVecGridBuffer(cl_command_queue q, cl_mem buffer, Grid *g)
{
    WriteBuffer(buffer, g->grid, g->param.total * sizeof(Vec), sizeof(GridParam), q);
}

void ReadFullGridBuffer(cl_command_queue q, cl_mem buffer, Grid *g)
{
    size_t off = 0;
    ReadBuffer(buffer, &g->param, sizeof(GridParam), off, q);
    off += sizeof(GridParam);
    ReadBuffer(buffer, g->grid, sizeof(Vec) * g->param.total, off, q);
    off += sizeof(Vec) * g->param.total;
    ReadBuffer(buffer, g->ani, sizeof(Anisotropy) * g->param.total, off, q);
    off += sizeof(Anisotropy) * g->param.total;
    ReadBuffer(buffer, g->pinning, sizeof(Pinning) * g->param.total, off, q);
    off += sizeof(Pinning) * g->param.total;
}

void ReadVecGridBuffer(cl_command_queue q, cl_mem buffer, Grid *g)
{
    ReadBuffer(buffer, g->grid, g->param.total * sizeof(Vec), sizeof(GridParam), q);
}
#endif