#ifndef __CONSTANTS_H
#define __CONSTANTS_H

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_E
#define M_E 2.7182818284590452354
#endif

#define HBAR 1.054571817e-34 // J*s
#define QE 1.602176634e-19 // C
#define MU_B 9.2740100783e-24 // J/T
#define MU_0 1.25663706212e-6 // N/A^2
#define KB 1.380649e-23 // J/K
#define NS 1.0e-9 // s
#define NANO 1.0e-9

#define UNUSED(x) ((void)x)
#define SIGN(x) ((x) > 0 ? 1.0: -1.0)

#define INCEPTION(M)

#define LOC(row, col, depth, rows, cols) ((col) + (row) * (cols) + (depth) * (cols) * (rows))
#define EPS 1.0e-12
#define CLOSE_ENOUGH(x, y, eps) ((SIGN((x) - (y)) * ((x) - (y))) <= (eps))

#ifndef OPENCL_COMPILATION
#define GLOBAL 
#define LOCAL 
#define PRIVATE 
#else
#define GLOBAL global
#define LOCAL local
#define PRIVATE private
#endif

#endif
