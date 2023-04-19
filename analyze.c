#include <funcs.h>
#include <grid.h>
#include <helpers.h>
#include <helpers_simulator.h>
//TODO: add support for windows api 
//https://learn.microsoft.com/pt-br/windows/win32/api/memoryapi/nf-memoryapi-mapviewoffile?redirectedfrom=MSDN
#include <sys/mman.h>

#define DEF "./output/grid_anim_dump.bin"

//TODO: parse command line argument (the right way)

int main(int argc, const char **argv)
{
    GridParam params = {0};
    GetGridParam("./input/input.in", &params);
    
    const char *file_path = "./output/integration_fly.bin";
    FILE *file = fopen(file_path, "rb");
    if (!file)
    {
        fprintf(stderr, "Could not open file %s: %s\n", file_path, strerror(errno));
        exit(1);
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *buffer = mmap(NULL, file_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
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
    Vec *grid      = ((Vec*)ptr);
    PBC pbc = params.pbc;

    size_t frame_size = rows * cols;

    printf("rows: %d cols: %d frames: %d cut: %d dt: %e lattice: %e\n", rows, cols, frames, cut, dt, lattice);

    
    FILE *out_data = fopen("./output/analyze_output.dat", "w");

    int row_stripes = 50;
    int col_stripes = 1;

    printf("y_stripes: %d x_stripes: %d\n", row_stripes, col_stripes);

    int rows_per_stripe = rows / row_stripes;
    int cols_per_stripe = cols / col_stripes;

    for (int t = 1; t < frames - 1; ++t)
    {
        Vec *gp, *gc, *gn;
        
        gc = &grid[frame_size * t];
        if (t == 0)
            gp = gc;
        else
            gp = &grid[frame_size * (t - 1)];

        if (t == frames - 1)
            gn = gc;
        else
            gn = &grid[frame_size * (t + 1)];


        for (int rs = 0; rs < row_stripes; ++rs)
        {
            for (int cs = 0; cs < col_stripes; ++cs)
            {
                Vec vel = {0};
                double charge_pr = 0.0;
                double charge_im = 0.0;
                for (int row = rs * rows_per_stripe; row < (rs + 1) * rows_per_stripe; ++row)
                {
                    for (int col = cs * cols_per_stripe; col < (cs + 1) * cols_per_stripe; ++col)
                    {
                        vel = VecAdd(vel, VelWeightedI(row * cols + col, gc, gp, gn, rows, cols, lattice, lattice, dt * cut, pbc));
                        charge_pr += ChargeI(row * cols + col, gc, rows, cols, pbc);
                        charge_im += ChargeI_old(row * cols + col, gc, rows, cols, lattice, lattice, pbc);
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

    munmap(buffer, file_size);
    return 0;
}

/*int main2(int argc, const char** argv)
{
    GridParam params = {0};
    GetGridParam("./input/input.in", &params);
    double J_abs = fabs(params.exchange);
    double a = params.lattice;

    StartParse("./input/input.in");
    int cut = GetValueInt("CUT", 10);
    double dtl = GetValueDouble("DT");
    double dt = dtl * HBAR / J_abs;
    double dtw = dt * (double)cut;
    EndParse();

    const char* file_name = DEF;
    if (argc > 1)
        file_name = argv[1];

    FILE *bin = fopen(file_name, "rb");
    if (!bin)
    {
        fprintf(stderr, "Could not open file %s: %s\n", file_name, strerror(errno));
        return 1;
    }

    int rows, cols, steps;
    fseek(bin, 0, SEEK_END);
    size_t file_size = ftell(bin);
    fseek(bin, 0, SEEK_SET);

    char* buffer = (char*)malloc(file_size);
    fread(buffer, file_size, 1, bin);
    char* ptr = buffer;

    rows = *((int*)ptr);
    ptr += sizeof(int);

    cols = *((int*)ptr);
    ptr += sizeof(int);

    steps = *((int*)ptr);
    ptr += sizeof(int);

    printf("Size: %zu Rows: %d Cols: %d Steps: %d\n", file_size, rows, cols, steps);

    Vec *grid = (Vec*)ptr;
    fclose(bin);

    int y_factor = 2;
    int x_factor = 1;
    int rows_per_stripe = rows / y_factor;
    int cols_per_stripe = cols / x_factor;

    Vec *vels = (Vec*)calloc(y_factor * x_factor, sizeof(Vec));
    double *charges = (double*)calloc(y_factor * x_factor, sizeof(double));

    FILE *out = fopen("./output/out_vel_analyze.out", "w");
    if (!out)
    {
        fprintf(stderr, "Could not open file %s: %s\n", "./output/out_vel_analyze.out", strerror(errno));
        return 1;
    }
    fprintf(out, "time,x_start,x_end,y_start,y_end,vx,vy\n");
    for (int t = 0; t < steps; ++t)
    {
        if (t % (steps / 10) == 0)
            printf("%.3f%%\n", (double)t / (double)steps * 100.0);

        if (t == 0 || t >= steps - 2)
            continue;
        
        memset(charges, 0, sizeof(double) * y_factor * x_factor);
        memset(vels, 0, sizeof(Vec) * y_factor * x_factor);
        for (int y_stripe = 0; y_stripe < y_factor; ++y_stripe)
        {
            for (int x_stripe = 0; x_stripe < x_factor; ++x_stripe)
            {
                for (int row = rows_per_stripe * y_stripe; row < rows_per_stripe * (y_stripe + 1); ++row)
                {
                    for (int col = cols_per_stripe * x_stripe; col < cols_per_stripe * (x_stripe + 1); ++col)
                    {
                        charges[y_stripe * x_factor + x_stripe] += ChargeI(row * cols + col, &grid[t * rows * cols], rows, cols, params.pbc);
                        vels[y_stripe * x_factor + x_stripe] = VecAdd(vels[y_stripe * x_factor + x_stripe], 
                                                               VelWeightedI(row * cols + col, 
                                                                            &grid[t * rows * cols], 
                                                                            &grid[(t - 1) * rows * cols], 
                                                                            &grid[(t + 1) * rows * cols], rows, cols, a, a, dtw, params.pbc));
                    }
                }
                vels[y_stripe * x_factor + x_stripe] = VecScalar(vels[y_stripe * x_factor + x_stripe], 1.0 / charges[y_stripe * x_factor + x_stripe]);
                if (fabs(charges[y_stripe * x_factor + x_stripe]) < 0.3)
                    vels[y_stripe * x_factor + x_stripe] = VecFromScalar(0.0);
                fprintf(out, "%e,%d,%d,%d,%d,%e,%e\n", (int)t * dtw, cols_per_stripe * x_stripe, cols_per_stripe * (x_stripe + 1),
                                                                    rows_per_stripe * y_stripe, rows_per_stripe * (y_stripe + 1),
                                                                    vels[y_stripe * x_factor + x_stripe].x, vels[y_stripe * x_factor + x_stripe].y);
            }
        }
    } 

    fclose(out);

    if (vels)
        free(vels);
    
    if (charges)
        free(charges);

    if (buffer)
        free(buffer);
    return 0;
}*/
