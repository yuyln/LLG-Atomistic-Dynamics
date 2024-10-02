#include "atomistic_simulation.h"
#include <stdint.h>
#include <float.h>

typedef struct {
    uint64_t idx;
    v3d m;
    double angle;
    double x, y, z;
} point;

typedef struct {
    point *items;
    uint64_t len;
    uint64_t cap;
} points;

int main(void) {
    const char *name = "test.vtk";
    grid g = {0};
    if (!grid_from_animation_bin("./integrate_evolution.dat", &g, -1))
        logging_log(LOG_FATAL, "Could not read integration file");

    points ps = {0};
    for (int64_t z = 0; z < g.gi.depth; ++z) {
        for (int dz = 0; dz <= 0; ++dz) {
            for (int64_t y = 0; y < g.gi.rows; ++y) {
                for (int dy = 0; dy <= 0; ++dy) {
                    for (int64_t x = 0; x < g.gi.cols; ++x) {
                        for (int dx = 0; dx <= 0; ++dx) {
                            point p = {0};
                            p.idx = ps.len;
                            p.m = V_AT(g.m, y, x, z, g.gi.rows, g.gi.cols);
                            p.angle = atan2(p.m.y, p.m.x);
                            p.x = x + 0.5 * dx - 0.01 * SIGN(dx);
                            p.y = y + 0.5 * dy - 0.01 * SIGN(dy);
                            p.z = z + 0.5 * dz - 0.01 * SIGN(dz);
                            da_append(&ps, p);
                        }
                    }
                }
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
