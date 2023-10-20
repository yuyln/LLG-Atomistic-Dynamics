#ifndef __GEN_LATTICE_UTILS_H
#define __GEN_LATTICE_UTILS_H
#include <math.h>
#include <stdio.h>

typedef struct {
    double x, y, z;
} v3d;
void create_skyrmion_bloch(v3d *g, int rows, int cols, int cx, int cy, int R, double P, double Q);
void create_skyrmion_neel(v3d *g, int rows, int cols, int cx, int cy, int R, double P, double Q);
void create_biskyrmion_bloch(v3d *g, int rows, int cols, int cx, int cy, int R, double P/*UNUSED*/, double Q, int dx, int dy);
void create_biskyrmion_neel(v3d *g, int rows, int cols, int cx, int cy, int R, double P/*UNUSED*/, double Q, int dx, int dy);
void create_triangular_neel_skyrmion_lattice(v3d *g, int rows, int cols, int R, int nx, double P, double Q);
void create_triangular_bloch_skyrmion_lattice(v3d *g, int rows, int cols, int R, int nx, double P, double Q);
void print_grid(FILE *f, v3d *v, int rows, int cols);
void dump_grid(FILE *f, v3d *v, int rows, int cols);

#endif //__GEN_LATTICE_UTILS_H
#define __GEN_LATTICE_UTILS_C

#ifdef __GEN_LATTICE_UTILS_C
void create_skyrmion_bloch(v3d *g, int rows, int cols, int cx, int cy, int R, double P, double Q) {
    double R2 = R * R;
    for (int i = cy - 2 * R; i < cy + 2 * R; ++i) {
        double dy = (double)i - cy;
        int il = i % rows;
        if (il < 0)
            il += rows;
        for (int j = cx - 2 * R; j < cx + 2 * R; ++j) {
            int jl = j % cols;
            if (jl < 0)
                jl += cols;

            double dx = (double)j - cx;
            double r2 = dx * dx + dy * dy;
            double r = sqrt(r2);
            if (r > (2.0 * R))
                continue;

            g[il * cols + jl].z = 2.0 * Q * (exp(-r2 / R2) - 0.5);

            if (r != 0) {
                g[il * cols + jl].x = -dy * P / r * (1.0 - fabs(g[il * cols + jl].z));
                g[il * cols + jl].y = dx * P / r * (1.0 - fabs(g[il * cols + jl].z));
            }
            else {
                g[il * cols + jl].x = 0.0;
                g[il * cols + jl].y = 0.0;
            }
        }
    }
}

void create_skyrmion_neel(v3d *g, int rows, int cols, int cx, int cy, int R, double P, double Q) {
    double R2 = R * R;
    for (int i = cy - 2 * R; i < cy + 2 * R; ++i) {
        double dy = (double)i - cy;
        int il = i % rows;
        if (il < 0)
            il += rows;
        for (int j = cx - 2 * R; j < cx + 2 * R; ++j) {
            int jl = j % cols;
            if (jl < 0)
                jl += cols;

            double dx = (double)j - cx;
            double r2 = dx * dx + dy * dy;
            double r = sqrt(r2);
            if (r > (1.5 * R))
                continue;

            g[il * cols + jl].z = 2.0 * Q * (exp(-r2 / R2) - 0.5);

            if (r != 0) {
                g[il * cols + jl].x = dx * P / r * (1.0 - fabs(g[il * cols + jl].z));
                g[il * cols + jl].y = dy * P / r * (1.0 - fabs(g[il * cols + jl].z));
            }
            else {
                g[il * cols + jl].x = 0.0;
                g[il * cols + jl].y = 0.0;
            }
        }
    }
}

void create_biskyrmion(v3d *g, int rows, int cols, int cx, int cy, int R, double P, double Q, double xi, int dx, int dy) {
    for (int yl = 0; yl < rows; ++yl) {
        double y = yl - cy;
        for (int xl = 0; xl < cols; ++xl) {
            double x = xl - cx;
            double yn = y - dy / 2.0;
            double yp = y + dy / 2.0;
            double xn = x - dx / 2.0;
            double xp = x + dx / 2.0;
            double phi = atan2(yn, xn) + atan2(yp, xp) + xi;
            double theta = 2.0 * atan2(1.0 / (R * R) * sqrt(xn * xn + yn * yn) * sqrt(xp * xp + yp * yp), 1.0);
            g[yl * cols + xl] = (v3d){.x = Q * cos(phi) * sin(theta),
                                      .y = Q * sin(phi) * sin(theta),
                                      .z = Q * cos(theta)};
        }
    }
}

void create_biskyrmion_bloch(v3d *g, int rows, int cols, int cx, int cy, int R, double P/*UNUSED*/, double Q, int dx, int dy) {
    create_biskyrmion(g, rows, cols, cx, cy, R, P, Q, 0.0, dx, dy);
}

void create_biskyrmion_neel(v3d *g, int rows, int cols, int cx, int cy, int R, double P/*UNUSED*/, double Q, int dx, int dy) {
    create_biskyrmion(g, rows, cols, cx, cy, R, P, Q, M_PI / 2.0, dx, dy);
}

void create_triangular_neel_skyrmion_lattice(v3d *g, int rows, int cols, int R, int nx, double P, double Q) {
    double Sl = ((double)cols - 2.0 * R * (double)nx) / (double)nx;
    double S = Sl + 2.0 * R;

    double yc = sqrt(3.0) / 4.0 * S;
    int j = 0;
    while (yc < rows) {
        for (int i = 0; i < nx; ++i) {
            double xc = S / 2.0 + i * S;
            if (j % 2)
                xc -= S / 2.0;
            create_skyrmion_neel(g, rows, cols, xc, yc, R, P, Q);
        }
        yc += sqrt(3.0) / 2.0 * S;
        ++j;
    }
}

void create_triangular_bloch_skyrmion_lattice(v3d *g, int rows, int cols, int R, int nx, double P, double Q) {
    double Sl = ((double)cols - 2.0 * R * (double)nx) / (double)nx;
    double S = Sl + 2.0 * R;

    double yc = sqrt(3.0) / 4.0 * S;
    int j = 0;
    while (yc < rows) {
        for (int i = 0; i < nx; ++i) {
            double xc = S / 2.0 + i * S;
            if (j % 2)
                xc -= S / 2.0;
            create_skyrmion_bloch(g, rows, cols, xc, yc, R, P, Q);
        }
        yc += sqrt(3.0) / 2.0 * S;
        ++j;
    }
}

void print_grid(FILE *f, v3d *v, int rows, int cols) {
    for (int row = rows - 1; row >= 1; --row) {
        for (int col = 0; col < cols - 1; ++col) {
            fprintf(f, "%.15f\t%.15f\t%.15f\t", v[row * cols + col].x, v[row * cols + col].y, v[row * cols + col].z);
        }
        int col = cols - 1;
        fprintf(f, "%.15f\t%.15f\t%.15f\n", v[row * cols + col].x, v[row * cols + col].y, v[row * cols + col].z);
    }

    int row = 0;

    for (int col = 0; col < cols - 1; ++col) {
        fprintf(f, "%.15f\t%.15f\t%.15f\t", v[row * cols + col].x, v[row * cols + col].y, v[row * cols + col].z);
    }
    int col = cols - 1;
    fprintf(f, "%.15f\t%.15f\t%.15f", v[row * cols + col].x, v[row * cols + col].y, v[row * cols + col].z);
}

void dump_grid(FILE *f, v3d *v, int rows, int cols) {
    fprintf(f, "BINARY");
    fwrite(&rows, sizeof(int), 1, f);
    fwrite(&cols, sizeof(int), 1, f);
    fwrite(v, rows * cols * sizeof(v3d), 1, f);
}

#endif //__GEN_LATTICE_UTILS_C
