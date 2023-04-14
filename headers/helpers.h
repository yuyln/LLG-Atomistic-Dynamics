#ifndef __HELP
#define __HELP

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <omp.h>

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

#if __STDC_VERSION__ > 201603L
#define mynodiscard [[nodiscard]]
#else
#define mynodiscard 
#endif

typedef struct
{
    double qA, qT, qV, T0;
    size_t inner_loop, outer_loop, print_param;
} GSAParam;

typedef struct GPU
{
    cl_platform_id *plats; size_t n_plats; int i_plat;
    cl_device_id *devs; size_t n_devs; int i_dev;
    cl_context ctx;
    cl_command_queue queue;
    cl_program program;
    Kernel *kernels; size_t n_kernels;
} GPU;

typedef struct Simulator
{
    size_t n_steps;
    double dt;
    size_t n_cpu;
    size_t write_cut;
    size_t write_vel_charge_cut;
    bool write_to_file, use_gpu, do_gsa, do_relax, doing_relax, do_integrate, write_human, write_on_fly, calculate_energy;
    GSAParam gsap;
    GPU gpu;
    Grid g_old;
    Grid g_new;
    cl_mem g_old_buffer, g_new_buffer;
    Vec *grid_out_file, *velxy_Ez, *pos_xy, *avg_mag, *chpr_chim;
} Simulator;

FILE* mfopen(const char* name, const char* mode, int exit_)
{
    FILE *f = fopen(name, mode);
    if (!f)
    {
        fprintf(stderr, "Could not open file %s: %s\n", name, strerror(errno));
        if (exit_) exit(exit_);
    }
    return f;
}


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
    {
        free(g->grid);
        g->grid = NULL;
    }

    if (g->ani)
    {
        free(g->ani);
        g->ani = NULL;
    }
    
    if (g->pinning)
    {
        free(g->pinning);
        g->pinning = NULL;
    }
    
    if (g->regions)
    {
        free(g->regions);
        g->regions = NULL;
    }
}

void CopyGrid(Grid *to, Grid *from)
{
    FreeGrid(to);
    memcpy(&to->param, &from->param, sizeof(GridParam));

    to->grid = (Vec*)calloc(from->param.total, sizeof(Vec));
    to->ani = (Anisotropy*)calloc(from->param.total, sizeof(Anisotropy));
    to->pinning = (Pinning*)calloc(from->param.total, sizeof(Pinning));
    to->regions = (RegionParam*)calloc(from->param.total, sizeof(RegionParam));

    memcpy(to->grid, from->grid, from->param.total * sizeof(Vec));
    memcpy(to->ani, from->ani, from->param.total * sizeof(Anisotropy));
    memcpy(to->pinning, from->pinning, from->param.total * sizeof(Pinning));
    memcpy(to->regions, from->regions, from->param.total * sizeof(RegionParam));
}

void CopyToExistingGrid(Grid *to, Grid *from)
{
    memcpy(to->grid, from->grid, from->param.total * sizeof(Vec));
    memcpy(to->ani, from->ani, from->param.total * sizeof(Anisotropy));
    memcpy(to->pinning, from->pinning, from->param.total * sizeof(Pinning));
    memcpy(to->regions, from->regions, from->param.total * sizeof(RegionParam));
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
    FILE* f = mfopen(path, "rb", 1);

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
    out.regions = (RegionParam*)calloc(out.param.total, sizeof(RegionParam));
    return out;
}

Grid InitGridRandom(int rows, int cols)
{
    srand(time(NULL));
    Grid ret = InitNullGrid();
    ret.param.rows = rows;
    ret.param.cols = cols;
    ret.param.total = rows * cols;
    ret.grid = InitVecGridRandom(rows, cols);
    ret.ani = (Anisotropy*)calloc(ret.param.total, sizeof(Anisotropy));
    ret.pinning = (Pinning*)calloc(ret.param.total, sizeof(Pinning));
    ret.regions = (RegionParam*)calloc(ret.param.total, sizeof(RegionParam));
    return ret;
}

size_t FindGridSize(const Grid* g)
{
    size_t param = sizeof(GridParam);
    size_t grid_vec = g->param.total * sizeof(Vec);
    size_t grid_pinning = g->param.total * sizeof(Pinning);
    size_t grid_ani = g->param.total * sizeof(Anisotropy);
    size_t grid_regions = g->param.total * sizeof(RegionParam);
    return param + grid_vec + grid_pinning + grid_ani + grid_regions;
}

void PrintVecGrid(FILE* f, Vec* v, int rows, int cols)
{
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
}

void PrintVecGridToFile(const char* path, Vec* v, int rows, int cols)
{
    FILE *f = mfopen(path, "wb", 1);

    PrintVecGrid(f, v, rows, cols);

    fclose(f);
}

void GetGridParam(const char* path, GridParam* g)
{
    StartParse(path);

    g->exchange = GetValueDouble("EXCHANGE");
    g->dm = GetValueDouble("DMI") * g->exchange;
    g->lattice = GetValueDouble("LATTICE");
    g->cubic_ani = GetValueDouble("CUBIC");
    g->lande = GetValueDouble("LANDE");
    g->avg_spin = GetValueDouble("SPIN");
    g->mu_s = g->lande * MU_B * g->avg_spin;
    g->alpha = GetValueDouble("ALPHA");
    g->gamma = GetValueDouble("GAMMA");

    if (GetValueInt("DM_TYPE", 10) > 1 || GetValueInt("DM_TYPE", 10) < 0)
    {
        fprintf(stderr, "Invalid DM\n");
        exit(1);
    }
    g->dm_type = GetValueInt("DM_TYPE", 10);


    if (GetValueInt("PBC_TYPE", 10) > 3 || GetValueInt("PBC_TYPE", 10) < 0)
    {
        fprintf(stderr, "Invalid PBC\n");
        exit(1);
    }
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
    WriteBuffer(buffer, g->regions, sizeof(RegionParam) * g->param.total, off, q);
    off += sizeof(RegionParam) * g->param.total;
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
    ReadBuffer(buffer, g->regions, sizeof(RegionParam) * g->param.total, off, q);
    off += sizeof(RegionParam) * g->param.total;
}

void ReadVecGridBuffer(cl_command_queue q, cl_mem buffer, Grid *g)
{
    ReadBuffer(buffer, g->grid, g->param.total * sizeof(Vec), sizeof(GridParam), q);
}


//@TODO: Different cut for velocity and charge files
void IntegrateSimulatorSingle(Simulator* s, Vec field, Current cur, const char* file_name)
{
    FILE *fly = mfopen(file_name, "wb", 1);
    if (s->write_on_fly && (!s->doing_relax))
    {
        fwrite(&s->g_old.param.rows, sizeof(int), 1, fly);
        fwrite(&s->g_old.param.cols, sizeof(int), 1, fly);
        int steps = (int)(s->n_steps / s->write_cut);
        fwrite(&steps, sizeof(int), 1, fly);
    }
    memset(s->velxy_Ez, 0, sizeof(Vec) * s->n_steps / s->write_vel_charge_cut);
    memset(s->pos_xy, 0, sizeof(Vec) * s->n_steps / s->write_vel_charge_cut);
    memset(s->avg_mag, 0, sizeof(Vec) * s->n_steps / s->write_vel_charge_cut);
    memset(s->chpr_chim, 0, sizeof(Vec) * s->n_steps / s->write_vel_charge_cut);
    
    for (size_t i = 0; i < s->n_steps; ++i)
    {
        if (i % (s->n_steps / 10) == 0)
        {
            printf("%.3f%%\n", 100.0 * (double)i / (double)s->n_steps);
            fflush(stdout);
        }
        double norm_time = (double)i * s->dt * (!s->doing_relax);
        size_t t = i / s->write_vel_charge_cut;
        for (size_t I = 0; I < s->g_old.param.total; ++I)
        {
            s->g_new.grid[I] = VecAdd(s->g_old.grid[I], StepI(I, &s->g_old, field, cur, s->dt, norm_time));
            GridNormalizeI(I, &s->g_new);

            if (i % s->write_vel_charge_cut == 0)
            {
                size_t x = I % s->g_old.param.cols;
                size_t y = (I - x) / s->g_old.param.cols;
                double charge_i = ChargeI(I, s->g_new.grid, s->g_old.param.rows, s->g_old.param.cols, s->g_old.param.lattice, s->g_old.param.lattice, s->g_old.param.pbc);
                double charge_i_old = ChargeI_old(I, s->g_new.grid, s->g_old.param.rows, s->g_old.param.cols, s->g_old.param.lattice, s->g_old.param.lattice, s->g_old.param.pbc);
                s->pos_xy[t].x += (double)x * s->g_old.param.lattice * charge_i;
                s->pos_xy[t].y += (double)y * s->g_old.param.lattice * charge_i;

                Vec vt = VelWeightedI(I, s->g_new.grid, s->g_old.grid, s->g_new.grid, s->g_old.param.rows, s->g_old.param.cols, 
                            s->g_old.param.lattice, s->g_old.param.lattice, 0.5 * s->dt * HBAR / fabs(s->g_old.param.exchange), s->g_old.param.pbc);

                s->velxy_Ez[t].x += vt.x;
                s->velxy_Ez[t].y += vt.y;

                s->avg_mag[t] = VecAdd(s->avg_mag[t], s->g_new.grid[I]);

                s->chpr_chim[t].x = charge_i;
                s->chpr_chim[t].y = charge_i_old;
            }
        }
        
        memcpy(s->g_old.grid, s->g_new.grid, sizeof(Vec) * s->g_old.param.total);
        if (i % s->write_vel_charge_cut == 0)
        {
            // Moved to export phase
            /*s->velxy_Ez_chargez[t].x /= s->velxy_Ez_chargez[t].z;
            s->velxy_Ez_chargez[t].y /= s->velxy_Ez_chargez[t].z;
            s->pos_xy[t].x /= s->velxy_Ez_chargez[t].z;
            s->pos_xy[t].y /= s->velxy_Ez_chargez[t].z;*/
            s->avg_mag[t] = VecScalar(s->avg_mag[t], 1.0 / (double)s->g_old.param.total);
        }

        if (i % s->write_cut == 0)
        {
            t = i / s->write_cut;
            if (s->write_to_file)
                memcpy(&s->grid_out_file[t * s->g_old.param.total], s->g_old.grid, sizeof(Vec) * s->g_old.param.total);
            if (s->write_on_fly && (!s->doing_relax))
                fwrite(s->g_old.grid, sizeof(Vec) * s->g_old.param.total, 1, fly);
        }
    }

    fclose(fly);
}

void IntegrateSimulatorMulti(Simulator* s, Vec field, Current cur, const char* file_name)
{
    FILE *fly = mfopen(file_name, "wb", 1);
    if (s->write_on_fly && (!s->doing_relax))
    {
        fwrite(&s->g_old.param.rows, sizeof(int), 1, fly);
        fwrite(&s->g_old.param.cols, sizeof(int), 1, fly);
        int steps = (int)(s->n_steps / s->write_cut);
        fwrite(&steps, sizeof(int), 1, fly);
    }

    memset(s->velxy_Ez, 0, sizeof(Vec) * s->n_steps / s->write_vel_charge_cut);
    memset(s->pos_xy, 0, sizeof(Vec) * s->n_steps / s->write_vel_charge_cut);
    memset(s->avg_mag, 0, sizeof(Vec) * s->n_steps / s->write_vel_charge_cut);
    memset(s->chpr_chim, 0, sizeof(Vec) * s->n_steps / s->write_vel_charge_cut);

    Vec *velxy_thread = (Vec*)(calloc(s->n_cpu, sizeof(Vec)));
    Vec *pos_xy_thread = (Vec*)(calloc(s->n_cpu, sizeof(Vec)));
    Vec *avg_mag_thread = (Vec*)(calloc(s->n_cpu, sizeof(Vec)));
    Vec *chpr_chim_thread = (Vec*)(calloc(s->n_cpu, sizeof(Vec)));

    for (size_t i = 0; i < s->n_steps; ++i)
    {
        double norm_time = (double)i * s->dt * (!s->doing_relax);
        size_t t = i / s->write_vel_charge_cut;

        if (i % s->write_vel_charge_cut == 0)
        {
            memset(velxy_thread, 0, sizeof(Vec) * s->n_cpu);
            memset(pos_xy_thread, 0, sizeof(Vec) * s->n_cpu);
            memset(avg_mag_thread, 0, sizeof(Vec) * s->n_cpu);
            memset(chpr_chim_thread, 0, sizeof(Vec) * s->n_cpu);
        }

        if (i % (s->n_steps / 10) == 0)
        {
            printf("%.3f%%\n", 100.0 * (double)i / (double)s->n_steps);
            fflush(stdout);
        }

        int I = 0; //For Visual Studio not complain
        #pragma omp parallel for num_threads(s->n_cpu)
        for (I = 0; I < s->g_old.param.total; ++I)
        {
            s->g_new.grid[I] = VecAdd(s->g_old.grid[I], StepI(I, &s->g_old, field, cur, s->dt, norm_time));
            GridNormalizeI(I, &s->g_new);

            if (i % s->write_vel_charge_cut == 0)
            {
                int nt = omp_get_thread_num();
                size_t x = I % s->g_old.param.cols;
                size_t y = (I - x) / s->g_old.param.cols;
                double charge_i = ChargeI(I, s->g_new.grid, s->g_old.param.rows, s->g_old.param.cols, s->g_old.param.lattice, s->g_old.param.lattice, s->g_old.param.pbc);
                double charge_i_old = ChargeI_old(I, s->g_new.grid, s->g_old.param.rows, s->g_old.param.cols, s->g_old.param.lattice, s->g_old.param.lattice, s->g_old.param.pbc);
                pos_xy_thread[nt].x += (double)x * s->g_old.param.lattice * charge_i;
                pos_xy_thread[nt].y += (double)y * s->g_old.param.lattice * charge_i;

                Vec vt = VelWeightedI(I, s->g_new.grid, s->g_old.grid, s->g_new.grid, s->g_old.param.rows, s->g_old.param.cols, 
                            s->g_old.param.lattice, s->g_old.param.lattice, 0.5 * s->dt * HBAR / fabs(s->g_old.param.exchange), s->g_old.param.pbc);
                velxy_thread[nt].x += vt.x;
                velxy_thread[nt].y += vt.y;

                avg_mag_thread[nt] = VecAdd(avg_mag_thread[nt], s->g_new.grid[I]);

                chpr_chim_thread[nt] = VecAdd(chpr_chim_thread[nt], VecFrom(charge_i, charge_i_old, 0.0));
            }
        }
        
        memcpy(s->g_old.grid, s->g_new.grid, sizeof(Vec) * s->g_old.param.total);
        if (i % s->write_vel_charge_cut == 0) 
        {
            for (size_t k = 0; k < s->n_cpu; ++k)
            {
                s->velxy_Ez[t] = VecAdd(s->velxy_Ez[t], velxy_thread[k]);
                s->pos_xy[t] = VecAdd(s->pos_xy[t], pos_xy_thread[k]);
                s->avg_mag[t] = VecAdd(s->avg_mag[t], avg_mag_thread[k]);
                s->chpr_chim[t] = VecAdd(s->chpr_chim[t], chpr_chim_thread[k]);
            }

            // Moved to export phase
            /*s->velxy_Ez_chargez[t].x /= s->velxy_Ez_chargez[t].z;
            s->velxy_Ez_chargez[t].y /= s->velxy_Ez_chargez[t].z;
            s->pos_xy[t].x /= s->velxy_Ez_chargez[t].z;
            s->pos_xy[t].y /= s->velxy_Ez_chargez[t].z;*/
            s->avg_mag[t] = VecScalar(s->avg_mag[t], 1.0 / (double)s->g_old.param.total);
        }
        
        if (i % s->write_cut == 0)
        {
            t = i / s->write_cut;
            if (s->write_to_file)
                memcpy(&s->grid_out_file[t * s->g_old.param.total], s->g_old.grid, sizeof(Vec) * s->g_old.param.total);
            if (s->write_on_fly && (!s->doing_relax))
                fwrite(s->g_old.grid, sizeof(Vec) * s->g_old.param.total, 1, fly);
        }
    }
    fclose(fly);
    free(velxy_thread);
    free(pos_xy_thread);
    free(avg_mag_thread);
    free(chpr_chim_thread);
}

void IntegrateSimulatorGPU(Simulator *s, Vec field, Current cur, const char* file_name)
{
    FILE *fly = mfopen(file_name, "wb", 1);
    if (s->write_on_fly && (!s->doing_relax))
    {
        fwrite(&s->g_old.param.rows, sizeof(int), 1, fly);
        fwrite(&s->g_old.param.cols, sizeof(int), 1, fly);
        int steps = (int)(s->n_steps / s->write_cut);
        fwrite(&steps, sizeof(int), 1, fly);
    }

    memset(s->velxy_Ez, 0, sizeof(Vec) * s->n_steps / s->write_vel_charge_cut);
    memset(s->pos_xy, 0, sizeof(Vec) * s->n_steps / s->write_vel_charge_cut);
    memset(s->avg_mag, 0, sizeof(Vec) * s->n_steps / s->write_vel_charge_cut);
    memset(s->chpr_chim, 0, sizeof(Vec) * s->n_steps / s->write_vel_charge_cut);

    SetKernelArg(s->gpu.kernels[2], 0, sizeof(cl_mem), &s->g_old_buffer);
    SetKernelArg(s->gpu.kernels[2], 1, sizeof(cl_mem), &s->g_new_buffer);

    SetKernelArg(s->gpu.kernels[3], 0, sizeof(cl_mem), &s->g_old_buffer);
    SetKernelArg(s->gpu.kernels[3], 1, sizeof(cl_mem), &s->g_new_buffer);
    SetKernelArg(s->gpu.kernels[3], 2, sizeof(Vec), &field);
    SetKernelArg(s->gpu.kernels[3], 3, sizeof(double), &s->dt);
    SetKernelArg(s->gpu.kernels[3], 4, sizeof(Current), &cur);

    // int cut = s->write_vel_charge_cut;
    SetKernelArg(s->gpu.kernels[3], 7, sizeof(int), &s->write_vel_charge_cut);
    SetKernelArg(s->gpu.kernels[3], 9, sizeof(int), &s->calculate_energy);

    size_t global = s->g_old.param.total;
    size_t local = gcd(global, 32);

    Vec* vxvy_Ez_avg_mag_chpr_chim = (Vec*)calloc(3 * s->g_old.param.total, sizeof(Vec));
    memset(vxvy_Ez_avg_mag_chpr_chim, 1, sizeof(Vec) * s->g_old.param.total * 3);
    memset(vxvy_Ez_avg_mag_chpr_chim, 0, sizeof(Vec) * s->g_old.param.total * 3);

    cl_mem vxvy_Ez_avg_mag_chpr_chim_buffer = CreateBuffer(3 * sizeof(Vec) * s->g_old.param.total, s->gpu.ctx, CL_MEM_READ_WRITE);
    WriteBuffer(vxvy_Ez_avg_mag_chpr_chim_buffer, vxvy_Ez_avg_mag_chpr_chim, 3 * sizeof(Vec) * s->g_old.param.total, 0, s->gpu.queue);

    SetKernelArg(s->gpu.kernels[3], 8, sizeof(cl_mem), &vxvy_Ez_avg_mag_chpr_chim_buffer);

    for (size_t i = 0; i < s->n_steps; ++i)
    {
        if (i % (s->n_steps / 100) == 0)
        {
            printf("%.3f%%\n", 100.0 * (double)i / (double)s->n_steps);
            fflush(stdout);
        }

        double norm_time = (double)i * s->dt * (!s->doing_relax);
        SetKernelArg(s->gpu.kernels[3], 5, sizeof(double), &norm_time);
        int t = i;
        SetKernelArg(s->gpu.kernels[3], 6, sizeof(int), &t);
        
        EnqueueND(s->gpu.queue, s->gpu.kernels[3], 1, NULL, &global, &local);
        EnqueueND(s->gpu.queue, s->gpu.kernels[2], 1, NULL, &global, &local);

        if (i % s->write_vel_charge_cut == 0)
        {
            size_t t = i / s->write_vel_charge_cut;

            /*
                So, reading the GPU this frequent is BAAAD
                If Visual Studio is right, this read call takes about 60%-80% of the function call (cumulative)
                The optimal way would be to create a large buffer on the GPU, write to that buffer via kernels, and
                after all the integration is finished, read that buffer. However, this is almost impossible without
                using too much memory.
                A typical lattice of 272x272 with 3000000 integration steps would generate a huge buffer
                3 * sizeof(double) * 272 * 272 * (3000000 / s->write_vel_charge_cut) * 2(velocity and avg_mag)
               =3 * 8 * 272 * 272 * (3000000 / 100) * 2 (values used in real simulations made by me)
               =106.5GB
                So, no, there is no way of doing a large buffer for all the integration steps
                What could work is doing a smaller buffer, and reading it in batches during the integration.
                This way instead of 3000000 steps, as used above, we use the amount os elements per batch in the buffer.
                3 * sizeof(double) * 272 * 272 * elementes per batch * 2
               =3 * 8 * 272 * 272 * 500 * 2
               =1.7GB
                So, this would be more reasonable to use. However, usually we run many programs at the same time (~40)
                and 1GB(+other memory needed, such as grid buffer and so on) per program would not fit on any RTX GPU,
                so, yeah, no, we cannot use that either.
                We are left with reading the buffers more frequent than what i would like,
                but it is what it is and it isn't what it isn't
            */
            ReadBuffer(vxvy_Ez_avg_mag_chpr_chim_buffer, vxvy_Ez_avg_mag_chpr_chim, 3 * sizeof(Vec) * s->g_old.param.total, 0, s->gpu.queue);
            for (size_t k = 0; k < s->g_old.param.total; ++k)
            {
                size_t x = k % s->g_old.param.cols;
                size_t y = (k - x) / s->g_old.param.cols;
                s->velxy_Ez[t].x += vxvy_Ez_avg_mag_chpr_chim[k].x;
                s->velxy_Ez[t].y += vxvy_Ez_avg_mag_chpr_chim[k].y;
		s->velxy_Ez[t].z += vxvy_Ez_avg_mag_chpr_chim[k].z;

                s->pos_xy[t].x += (double)x * s->g_old.param.lattice * vxvy_Ez_avg_mag_chpr_chim[2 * s->g_old.param.total + k].x;
                s->pos_xy[t].y += (double)y * s->g_old.param.lattice * vxvy_Ez_avg_mag_chpr_chim[2 * s->g_old.param.total + k].x;

                s->avg_mag[t] = VecAdd(s->avg_mag[t], vxvy_Ez_avg_mag_chpr_chim[s->g_old.param.total + k]);

                s->chpr_chim[t] = VecAdd(s->chpr_chim[t], vxvy_Ez_avg_mag_chpr_chim[2 * s->g_old.param.total + k]);
            }

            // Moved to export phase
            /*s->velxy_Ez_chargez[t].x /= s->velxy_Ez_chargez[t].z;
            s->velxy_Ez_chargez[t].y /= s->velxy_Ez_chargez[t].z;
            s->pos_xy[t].x /= s->velxy_Ez_chargez[t].z;
            s->pos_xy[t].y /= s->velxy_Ez_chargez[t].z;*/
            s->avg_mag[t] = VecScalar(s->avg_mag[t], 1.0 / (double)s->g_old.param.total);
        }
        
        if (i % s->write_cut == 0)
        {
            size_t t = i / s->write_cut;
            if (s->write_to_file)
            {
                ReadVecGridBuffer(s->gpu.queue, s->g_old_buffer, &s->g_old);
                memcpy(&s->grid_out_file[t * s->g_old.param.total], s->g_old.grid, sizeof(Vec) * s->g_old.param.total);
            }
            if (s->write_on_fly && (!s->doing_relax))
            {
                ReadVecGridBuffer(s->gpu.queue, s->g_old_buffer, &s->g_old);
                fwrite(s->g_old.grid, sizeof(Vec) * s->g_old.param.total, 1, fly);
            }
        }
    }
    ReadVecGridBuffer(s->gpu.queue, s->g_old_buffer, &s->g_old);
    PrintCLError(stderr, clReleaseMemObject(vxvy_Ez_avg_mag_chpr_chim_buffer), "Could not release vxvy_Ez_avg_mag_chpr_chim_buffer obj");
    if (vxvy_Ez_avg_mag_chpr_chim)
        free(vxvy_Ez_avg_mag_chpr_chim);

    fclose(fly);
}

void IntegrateSimulator(Simulator *s, Vec field, Current cur, const char* file_name)
{
    if (s->use_gpu)
        IntegrateSimulatorGPU(s, field, cur, file_name);
    else if (s->n_cpu > 1)
        IntegrateSimulatorMulti(s, field, cur, file_name);
    else
        IntegrateSimulatorSingle(s, field, cur, file_name);
}

double NormCurToReal(double density, GridParam params)
{
    return 2.0 * QE * params.avg_spin * fabs(params.exchange) * density / (params.lattice * params.lattice * HBAR);
}

double RealCurToNorm(double density, GridParam params)
{
    return params.lattice * params.lattice * HBAR * density / (2.0 * QE * params.avg_spin * fabs(params.exchange));
}

double LatticeCharge(Vec *g, int rows, int cols, double dx, double dy, PBC pbc)
{
    double ret = 0.0;
    for (size_t i = 0; i < (size_t)rows * cols; ++i)
    {
        #ifdef INTERP
            ret += ChargeInterpI(i, g, rows, cols, dx, dy, pbc, INTERP);
        #else
            ret += ChargeI(i, g, rows, cols, dx, dy, pbc);
        #endif
    }
    return ret;
}

void CreateSkyrmionBloch(Vec *g, int rows, int cols, int cx, int cy, int R, double P, double Q)
{
    double R2 = R * R;
    for (int i = cy - 2 * R; i < cy + 2 * R; ++i)
    {
        double dy = (double)i - cy;
        int il = i % rows;
        if (il < 0) il += rows;
        for (int j = cx - 2 * R; j < cx + 2 * R; ++j)
        {
            int jl = j % cols;
            if (jl < 0) jl += cols;

            double dx = (double)j - cx;
            double r2 = dx * dx + dy * dy;
            double r = sqrt(r2);
            if (r > (2.0 * R))
                continue;
            
            g[il * cols + jl].z = 2.0 * Q * (exp(-r2 / R2) - 0.5);
        
            if (r != 0)
            {
                g[il * cols + jl].x = -dy * P / r * (1.0 - fabs(g[il * cols + jl].z));
                g[il * cols + jl].y = dx * P / r * (1.0 - fabs(g[il * cols + jl].z));
            }
            else
            {
                g[il * cols + jl].x = 0.0;
                g[il * cols + jl].y = 0.0;
            }
        }
    }
}

void CreateSkyrmionNeel(Vec *g, int rows, int cols, int cx, int cy, int R, double P, double Q)
{
    double R2 = R * R;
    for (int i = cy - 2 * R; i < cy + 2 * R; ++i)
    {
        double dy = (double)i - cy;
        int il = i % rows;
        if (il < 0) il += rows;
        for (int j = cx - 2 * R; j < cx + 2 * R; ++j)
        {
            int jl = j % cols;
            if (jl < 0) jl += cols;

            double dx = (double)j - cx;
            double r2 = dx * dx + dy * dy;
            double r = sqrt(r2);
            if (r > (2.0 * R))
                continue;
            
            g[il * cols + jl].z = 2.0 * Q * (exp(-r2 / R2) - 0.5);
        
            if (r != 0)
            {
                g[il * cols + jl].x = dx * P / r * (1.0 - fabs(g[il * cols + jl].z));
                g[il * cols + jl].y = dy * P / r * (1.0 - fabs(g[il * cols + jl].z));
            }
            else
            {
                g[il * cols + jl].x = 0.0;
                g[il * cols + jl].y = 0.0;
            }
        }
    }
}

void CreateBimeron(Vec *g, int rows, int cols, int cx, int cy, int r, double Q, double P)
{
    double R2 = r * r;
    for (int i = 0; i < rows; ++i)
    {
        double dy = (double)i - cy;
        for (int j = 0; j < cols; ++j)
        {
            double dx = (double)j - cx;
            double r2 = dx * dx + dy * dy;
            double r = sqrt(r2);
            
            g[i * cols + j].x = 2.0 * P * (exp(-r2 / R2) - 0.5);
        
            if (r != 0)
            {
                g[i * cols + j].z = dx * Q / r * (1.0 - fabs(g[i * cols + j].x));
                g[i * cols + j].y = dy * Q / r * (1.0 - fabs(g[i * cols + j].x));
            }
            else
            {
                g[i * cols + j].z = 0.0;
                g[i * cols + j].y = 0.0;
            }
        }
    }
}

Vec ChargeCenter(Vec *g, int rows, int cols, double dx, double dy, PBC pbc)
{
    Vec ret = VecFromScalar(0.0);
    double total_charge = 0.0;
    for (size_t I = 0; I < (size_t)rows * cols; ++I)
    {
        int col = I % cols;
        int row = (I - col) / cols;
        double local_charge;
        #ifdef INTERP
            local_charge = ChargeInterpI(I, g, rows, cols, dx, dy, pbc, INTERP);
        #else
            local_charge = ChargeI(I, g, rows, cols, dx, dy, pbc);
        #endif
        total_charge += local_charge;
        ret.x += col * local_charge;
        ret.y += row * local_charge;
    }
    return VecScalar(ret, 1.0 / total_charge);
}

Vec TotalVelocity(Vec *current, Vec *before, Vec *after, int rows, int cols, double dx, double dy, double dt, PBC pbc)
{
    Vec ret = VecFromScalar(0.0);
    for (size_t I = 0; I < (size_t)(rows * cols); ++I)
        ret = VecAdd(ret, VelI(I, current, before, after, rows, cols, dx, dy, dt, pbc));
    return VecScalar(ret, dx * dy);
}

Vec TotalVelocityW(Vec *current, Vec *before, Vec *after, int rows, int cols, double dx, double dy, double dt, PBC pbc)
{
    Vec ret = VecFromScalar(0.0);
    for (size_t I = 0; I < (size_t)(rows * cols); ++I)
        ret = VecAdd(ret, VelWeightedI(I, current, before, after, rows, cols, dx, dy, dt, pbc));
    return ret;
}

void WriteSimulatorSimulation(const char* root_path, Simulator* s)
{
    // char *out_grid_anim;
    // size_t out_grid_anim_size = snprintf(NULL, 0, "%s_grid.out", root_path) + 1;
    // out_grid_anim = (char*)calloc(out_grid_anim_size, 1);
    // snprintf(out_grid_anim, out_grid_anim_size, "%s_grid.out", root_path);
    // out_grid_anim[out_grid_anim_size - 1] = '\0';

    // char *out_cm_charge_anim;
    // size_t out_cm_charge_anim_size = snprintf(NULL, 0, "%s_cm_charge.out", root_path) + 1;
    // out_cm_charge_anim = (char*)calloc(out_cm_charge_anim_size, 1);
    // snprintf(out_cm_charge_anim, out_cm_charge_anim_size, "%s_cm_charge.out", root_path);
    // out_cm_charge_anim[out_cm_charge_anim_size - 1] = '\0';

    char *out_charge;
    size_t out_charge_size = snprintf(NULL, 0, "%s_charge.out", root_path) + 1;
    out_charge = (char*)calloc(out_charge_size, 1);
    snprintf(out_charge, out_charge_size, "%s_charge.out", root_path);
    out_charge[out_charge_size - 1] = '\0';

    char *out_velocity;
    size_t out_velocity_size = snprintf(NULL, 0, "%s_velocity.out", root_path) + 1;
    out_velocity = (char*)calloc(out_velocity_size, 1);
    snprintf(out_velocity, out_velocity_size, "%s_velocity.out", root_path);
    out_velocity[out_velocity_size - 1] = '\0';

    char *out_pos_xy;
    size_t out_pos_xy_size = snprintf(NULL, 0, "%s_pos_xy.out", root_path) + 1;
    out_pos_xy = (char*)calloc(out_pos_xy_size, 1);
    snprintf(out_pos_xy, out_pos_xy_size, "%s_pos_xy.out", root_path);
    out_pos_xy[out_pos_xy_size - 1] = '\0';

    char *out_avg_mag;
    size_t out_avg_mag_size = snprintf(NULL, 0, "%s_avg_mag.out", root_path) + 1;
    out_avg_mag = (char*)calloc(out_avg_mag_size, 1);
    snprintf(out_avg_mag, out_avg_mag_size, "%s_avg_mag.out", root_path);
    out_avg_mag[out_avg_mag_size - 1] = '\0';

    char *out_energy;
    size_t out_energy_size = snprintf(NULL, 0, "%s_energy.out", root_path) + 1;
    out_energy = (char*)calloc(out_energy_size, 1);
    snprintf(out_energy, out_energy_size, "%s_energy.out", root_path);
    out_energy[out_energy_size - 1] = '\0';

    // FILE* charge_anim = mfopen(out_cm_charge_anim, "w", 0);
    // FILE* grid_anim = mfopen(out_grid_anim, "w", 0);
    FILE *charge_total = mfopen(out_charge, "w", 0);
    FILE *velocity_total = mfopen(out_velocity, "w", 0);
    FILE *pos_xy = mfopen(out_pos_xy, "w", 0);
    FILE *avg_mag = mfopen(out_avg_mag, "w", 0);
    FILE *energy = mfopen(out_energy, "w", 0);

    // free(out_grid_anim);
    // free(out_cm_charge_anim);
    free(out_charge);
    free(out_velocity);
    free(out_pos_xy);
    free(out_avg_mag);
    free(out_energy);

    double J_abs = fabs(s->g_old.param.exchange);
    printf("Writing charges related output\n");
    for (size_t i = 0; i < s->n_steps; ++i)
    {
        if (i % (s->n_steps / 10) == 0)
            printf("%.3f%%\n", (double)i / (double)s->n_steps * 100.0);
        if (i % s->write_vel_charge_cut)
            continue;

        size_t t = i / s->write_vel_charge_cut;
        double vx, vy, chpr, chim;
        vx = s->velxy_Ez[t].x;
        vy = s->velxy_Ez[t].y;
        chpr = s->chpr_chim[t].x;
        chim = s->chpr_chim[t].y;
        fprintf(velocity_total, "%e\t%e\t%e\t%e\t%e\n", (double)i * s->dt * HBAR / J_abs, vx / chpr, vy / chpr, vx / chim, vy / chim);
        fprintf(charge_total, "%e\t%e\t%e\n", (double)i * s->dt * HBAR / J_abs, chpr, chim);
        fprintf(pos_xy, "%e\t%e\t%en", (double)i * s->dt * HBAR / J_abs, s->pos_xy[t].x / chpr, s->pos_xy[t].y / chpr);
        fprintf(avg_mag, "%e\t%e\t%e\t%e\n", (double)i * s->dt * HBAR / J_abs, s->avg_mag[t].x, s->avg_mag[t].y, s->avg_mag[t].z);
	fprintf(energy, "%e\t%e\n", (double)i * s->dt * HBAR / J_abs, s->velxy_Ez[t].z);
        // Vec charge_center = ChargeCenter(&s->grid_out_file[t * s->g_old.param.total], s->g_old.param.rows, s->g_old.param.cols, s->g_old.param.lattice, s->g_old.param.lattice, s->g_old.param.pbc);
        // fprintf(charge_anim, "%e\t%e\t%e\n", (double)i * s->dt * HBAR / J_abs, charge_center.x * s->g_old.param.lattice, charge_center.y * s->g_old.param.lattice);


        // double charge = LatticeCharge(&s->grid_out_file[t * s->g_old.param.total], s->g_old.param.rows, s->g_old.param.cols, s->g_old.param.lattice, s->g_old.param.lattice, s->g_old.param.pbc);
        // fprintf(charge_total, "%e\t%e\n", (double)i * s->dt * HBAR / J_abs, charge);

        // if (t == 0 || t >= (s->n_steps / s->write_cut - 2))
        //     continue;
        // Vec vel = TotalVelocityW(&s->grid_out_file[t       * s->g_old.param.total], 
        //                          &s->grid_out_file[(t - 1) * s->g_old.param.total],
        //                          &s->grid_out_file[(t + 1) * s->g_old.param.total], 
        //                          s->g_old.param.rows, s->g_old.param.cols, s->g_old.param.lattice, s->g_old.param.lattice, 
        //                          (double)s->write_cut * s->dt * HBAR / J_abs, s->g_old.param.pbc);
                                 
        // fprintf(velocity_total, "%e\t%e\t%e\n", (double)i * s->dt * HBAR / J_abs, vel.x / charge, vel.y / charge);
    } 
    printf("Done writing charges related output\n");
    // fclose(charge_anim);
    fclose(charge_total);
    fclose(velocity_total);
    fclose(pos_xy);
    fclose(avg_mag);
    fclose(energy);

    if (!(s->write_to_file && s->write_human))
        return;

    // printf("Writing grid output\n");
    // for (size_t i = 0; i < s->n_steps && s->write_to_file && s->write_human; ++i)
    // {
    //     if (i % (s->n_steps / 10) == 0)
    //         printf("%.3f%%\n", (double)i / (double)s->n_steps * 100.0);
    //     if (i % s->write_cut)
    //         continue;
    //     size_t t = i / s->write_cut;
    //     PrintVecGrid(grid_anim, &s->grid_out_file[t * s->g_old.param.total], s->g_old.param.rows, s->g_old.param.cols);
    //     fprintf(grid_anim, "\n");
    // }
    // printf("Done writing grid output\n");

    // fclose(grid_anim);

}

void DumpGrid(const char* file_path, Vec* g, int rows, int cols)
{
    FILE *f = mfopen(file_path, "w", 1);
    fwrite(&rows, sizeof(int), 1, f);
    fwrite(&cols, sizeof(int), 1, f);
    fwrite(g, sizeof(Vec) * rows * cols, 1, f);
    fclose(f);
}

void DumpGridCharge(const char* file_path, Vec* g, int rows, int cols, double dx, double dy, PBC pbc)
{
    FILE *f = mfopen(file_path, "w", 1);
    double* charge = (double*)calloc(rows * cols, sizeof(double));
    for (size_t I = 0; I < (size_t)(rows * cols); ++I)
        charge[I] = ChargeI(I, g, rows, cols, dx, dy, pbc);
    
    fwrite(&rows, sizeof(int), 1, f);
    fwrite(&cols, sizeof(int), 1, f);
    fwrite(charge, sizeof(double) * rows * cols, 1, f);

    free(charge);
    fclose(f);
}

void CreateNeelTriangularLattice(Vec *g, int rows, int cols, int R, int nx, double P, double Q)
{
    double Sl = ((double)cols - 2.0 * R * (double)nx) / (double)nx;
    double S = Sl + 2.0 * R;
    
    double yc = sqrt(3.0) / 4.0 * S;
    int j = 0;
    while (yc < rows) {
        for (int i = 0; i < nx; ++i) {
            double xc = S / 2.0 + i * S;
            if (j % 2) xc -= S / 2.0;
            CreateSkyrmionNeel(g, rows, cols, xc, yc, R, P, Q);
        }
        yc += sqrt(3.0) / 2.0 * S;
        ++j;
    }
}

void CreateBlochTriangularLattice(Vec *g, int rows, int cols, int R, int nx, double P, double Q)
{
    double Sl = ((double)cols - 2.0 * R * (double)nx) / (double)nx;
    double S = Sl + 2.0 * R;
    
    double yc = sqrt(3.0) / 4.0 * S;
    int j = 0;
    while (yc < rows) {
        for (int i = 0; i < nx; ++i) {
            double xc = S / 2.0 + i * S;
            if (j % 2) xc -= S / 2.0;
            CreateSkyrmionBloch(g, rows, cols, xc, yc, R, P, Q);
        }
        yc += sqrt(3.0) / 2.0 * S;
        ++j;
    }
}
#endif
