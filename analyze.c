#include "./headers/vec.h"
#include "./headers/funcs.h"
#include "./headers/grid.h"
#include "./headers/helpers.h"
#include "./headers/helpers_simulator.h"
#include <stdint.h>

#ifdef WIN
#include <windows.h>
HANDLE file_h = NULL;
HANDLE file_map_h = NULL;
void *map_view = NULL;
unsigned long lerror = 0;
#define CHECK_ERROR() do { lerror = GetLastError(); if (lerror != 0) printf("ERROR: %lu\n", GetLastError()); } while(0)

void *memory_map(uint64_t s) {
    uint32_t size_l, size_h;
    //uint32_t size_l = (uint32_t)(s & 0x00000000ffffffff);
    //uint32_t size_h = (uint32_t)(s & 0xffffffff00000000);
    //size_l = *(uint32_t*)&s;
    //size_h = *(uint32_t*)((uint32_t*)&s + 1);
    uint32_t S[2] = {0};
    *(uint64_t*)S = s;
    size_l = S[0];
    size_h = S[1];

    file_h = CreateFileA("analyze.tmp", GENERIC_READ | GENERIC_WRITE, 0, NULL, CREATE_NEW, FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, NULL);
    CHECK_ERROR();


    file_map_h = CreateFileMappingA(file_h, NULL, PAGE_READWRITE, size_h, size_l, NULL);
    CHECK_ERROR();

    map_view = MapViewOfFile(file_map_h, FILE_MAP_ALL_ACCESS, 0, 0, s);
    CHECK_ERROR();

    return map_view;
}

void memory_unmap(void *buffer, uint64_t s) {
    if (!UnmapViewOfFile(buffer))
        CHECK_ERROR();

    if (!CloseHandle(file_map_h))
        CHECK_ERROR();

    if (!CloseHandle(file_h))
        CHECK_ERROR();
}
#else
#include <sys/mman.h>

void *memory_map(uint64_t s) {
    return mmap(NULL, s, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
}

void memory_unmap(void *buffer, uint64_t s) {
    munmap(buffer, s);
}

#endif

#define DEF "./output/integration_fly.bin"

//TODO: parse command line argument (the right way)

int main(int argc, const char **argv) {
    grid_param_t params = {0};
    find_grid_param_path("./input/input.in", &params);
    
    const char *file_path = "./output/integration_fly.bin";
    FILE *file = fopen(file_path, "rb");
    if (!file) {
        fprintf(stderr, "Could not open file %s: %s\n", file_path, strerror(errno));
        exit(1);
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *buffer = memory_map(file_size);
    fread(buffer, file_size, 1, file);
    char *ptr = buffer;
    fclose(file);

    int rows       = *((int*)ptr);
    ptr += 4;
    int cols       = *((int*)ptr);
    ptr += 4;
    int frames     = *((int*)ptr);
    ptr += 4;
    int cut        = *((int*)ptr);
    ptr += 4;
    double dt      = *((double*)ptr);
    ptr += 8;
    double lattice = *((double*)ptr);
    ptr += 8;
    v3d *grid      = ((v3d*)ptr);
    pbc_t pbc = params.pbc;

    uint64_t frame_size = rows * cols;

    printf("rows: %d cols: %d frames: %d cut: %d dt: %e lattice: %e\n", rows, cols, frames, cut, dt, lattice);

    
    FILE *out_data = fopen("./output/analyze_output.dat", "w");

    int row_stripes = 1;
    int col_stripes = 1;

    printf("y_stripes: %d x_stripes: %d\n", row_stripes, col_stripes);

    int rows_per_stripe = rows / row_stripes;
    int cols_per_stripe = cols / col_stripes;

    for (int t = 1; t < frames - 1; ++t) {
        v3d *gp, *gc, *gn;
        
        gc = &grid[frame_size * t];
        gp = &grid[frame_size * (t - 1)];
        gn = &grid[frame_size * (t + 1)];


        for (int rs = 0; rs < row_stripes; ++rs) {
            for (int cs = 0; cs < col_stripes; ++cs) {
                v3d vel = {0};
                double charge_pr = 0.0;
                double charge_im = 0.0;
                for (int row = rs * rows_per_stripe; row < (rs + 1) * rows_per_stripe; ++row) {
                    for (int col = cs * cols_per_stripe; col < (cs + 1) * cols_per_stripe; ++col) {
                        vel = v3d_add(vel, velocity_weighted(row * cols + col, gc, gp, gn, rows, cols, lattice, lattice, dt * cut, pbc));
                        charge_pr += charge(row * cols + col, gc, rows, cols, pbc);
                        charge_im += charge_old(row * cols + col, gc, rows, cols, lattice, lattice, pbc);
                    }
                }
                fprintf(out_data, "%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\n", t * dt * cut, cs * cols_per_stripe * lattice,
                                                                          rs * rows_per_stripe * lattice, (cs + 1) * cols_per_stripe * lattice,
                                                                          (rs + 1) * rows_per_stripe * lattice, vel.x / charge_pr, vel.y / charge_pr,
                                                                          charge_pr, charge_im);
            }
        }
    }

    fclose(out_data);

    memory_unmap(buffer, file_size);
    return 0;
}
