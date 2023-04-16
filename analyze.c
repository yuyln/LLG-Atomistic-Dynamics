#include <helpers.h>
#include <gsa.h>
#include <helpers_simulator.h>
#define DEF "./output/grid_anim_dump.bin"

int main(int argc, const char** argv)
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
}
