#ifndef __GEN_LATTICE_UTILS_H
#define __GEN_LATTICE_UTILS_H
#include <math.h>

typedef struct {
    double x, y, z;
} v3d;
void create_skyrmion_bloch(v3d *g, int rows, int cols, int cx, int cy, int R, double P, double Q);
void create_skyrmion_neel(v3d *g, int rows, int cols, int cx, int cy, int R, double P, double Q);
void create_triangular_neel_skyrmion_lattice(v3d *g, int rows, int cols, int R, int nx, double P, double Q);
void create_triangular_bloch_skyrmion_lattice(v3d *g, int rows, int cols, int R, int nx, double P, double Q);
void print_grid(FILE *f, v3d *v, int rows, int cols);

#endif //__GEN_LATTICE_UTILS_H

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

#endif //__GEN_LATTICE_UTILS_C
