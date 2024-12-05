#include "atomistic_simulation.h"
#include <stdint.h>
#include <float.h>

typedef struct {
    uint64_t idx;
    v3d m;
    double angle;
    double x, y, z;
    v3d field;
} point;

typedef struct {
    point *items;
    uint64_t len;
    uint64_t cap;
} points;

int main(void) {
    const char *name = "test.vtk";
    grid g = {0};
    if (!grid_from_file("/home/jose/initial_conditions/hopfion_3d.bin", &g))
        logging_log(LOG_FATAL, "Could not read integration file");

    p_id = 1;
    gpu_cl gpu = gpu_cl_init(NULL, NULL, NULL, NULL, NULL);
    integrate_params ip = integrate_params_init();
    integrate_context ctx = integrate_context_init(&g, &gpu, ip);
    integrate_step(&ctx);
    integrate_get_info(&ctx);

    points ps = {0};
    for (int64_t z = 0; z < g.gi.depth; ++z) {
	for (int64_t y = 0; y < g.gi.rows; ++y) {
	    for (int64_t x = 0; x < g.gi.cols; ++x) {
		point p = {0};
		p.idx = ps.len;
		p.m = V_AT(g.m, y, x, z, g.gi.rows, g.gi.cols);
		p.angle = atan2(p.m.y, p.m.x);
		p.x = x;
		p.y = y;
		p.z = z;
		V_AT(ctx.info, y, x, z, g.gi.rows, g.gi.cols).magnetic_field_lattice = v3d_scalar(V_AT(ctx.info, y, x, z, g.gi.rows, g.gi.cols).magnetic_field_lattice, QE / (4.0 * M_PI * HBAR));
		p.field.x = V_AT(ctx.info, y, x, z, g.gi.rows, g.gi.cols).magnetic_field_lattice.x;
		p.field.y = V_AT(ctx.info, y, x, z, g.gi.rows, g.gi.cols).magnetic_field_lattice.y;
		p.field.z = V_AT(ctx.info, y, x, z, g.gi.rows, g.gi.cols).magnetic_field_lattice.z;
		da_append(&ps, p);
	    }
        }
    }
    
    FILE *f = mfopen(name, "wb");
    fprintf(f, "# vtk DataFile Version 2.0\n");
    fprintf(f, "Lattice\n");
    fprintf(f, "ASCII\n");
    fprintf(f, "DATASET STRUCTURED_GRID\n");
    fprintf(f, "DIMENSIONS %u %u %u\n", 1 * g.gi.cols, 1 * g.gi.rows, 1 * g.gi.depth);

    fprintf(f, "POINTS %lu float\n", ps.len);
    for (uint64_t i = 0; i < ps.len; ++i) {
        fprintf(f, "%f %f %f\n", ps.items[i].x, ps.items[i].y, ps.items[i].z);
    }
    fprintf(f, "\n");

    fprintf(f, "POINT_DATA %lu\n", ps.len);
    fprintf(f, "VECTORS spins float\n");
    for (uint64_t i = 0; i < ps.len; ++i)
        fprintf(f, "%f %f %f\n", ps.items[i].m.x, ps.items[i].m.y, ps.items[i].m.z);
    fprintf(f, "\n");

    fprintf(f, "VECTORS field float\n");
    double max_f = -FLT_MAX;
    for (uint64_t i = 0; i < ps.len; ++i) {
        fprintf(f, "%f %f %f\n", ps.items[i].field.x, ps.items[i].field.y, ps.items[i].field.z);
	max_f = fabs(ps.items[i].field.x) > max_f? fabs(ps.items[i].field.x): max_f;
	max_f = fabs(ps.items[i].field.y) > max_f? fabs(ps.items[i].field.y): max_f;
	max_f = fabs(ps.items[i].field.z) > max_f? fabs(ps.items[i].field.z): max_f;
    }
    logging_log(LOG_INFO, "%.15e", max_f);
    fprintf(f, "\n");

    fprintf(f, "FIELD mz_field 1\n");
    fprintf(f, "mz_table 1 %lu float\n", ps.len);
    for (uint64_t i = 0; i < ps.len; ++i)
        fprintf(f, "%f\n", ps.items[i].m.z);
    fprintf(f, "\n");

    fprintf(f, "FIELD angle_field 1\n");
    fprintf(f, "angle 1 %lu float\n", ps.len);
    for (uint64_t i = 0; i < ps.len; ++i)
        fprintf(f, "%f\n", ps.items[i].angle);
    fprintf(f, "\n");

    mfclose(f);
    return 0;
}
