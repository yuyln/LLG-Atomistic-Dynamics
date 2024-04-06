#include <stdint.h>
#include <stdlib.h>
#define __PROFILER_IMPLEMENTATION
#include "atomistic_simulation.h"

//@TODO: PROPER ERROR CHECKING URGENT!!!
//@TODO: tools for generating defects
int main(void) {
    int stripe_size = 20;
    unsigned int rows = 64;
    unsigned int cols = rows * 2;

    double lattice = 0.5e-9;
    double alpha = 0.3;
    double J = 1.0e-3 * QE;
    double dm = 0.18 * J;
    double ani = 0.02 * J;

    grid g = grid_init(rows, cols);
    grid_set_lattice(&g, lattice);
    grid_set_alpha(&g, alpha);
    grid_set_exchange(&g, J);
    grid_set_anisotropy(&g, (anisotropy){.dir=v3d_c(0.0, 0.0, 1.0), .ani = ani});
    double mu = g.gp->mu;

    dm_interaction default_dm = (dm_interaction){.dmv_down = v3d_c(-dm, 0.0, 0.0),
                                                 .dmv_up = v3d_c(dm, 0.0, 0.0),
                                                 .dmv_left = v3d_c(0.0, dm, 0.0),
                                                 .dmv_right = v3d_c(0.0, -dm, 0.0)};

    grid_set_dm(&g, default_dm);

    for (unsigned int r = 0; r < rows; ++r)
        for (unsigned int c = 0; c < cols; ++c)
            g.m[r * cols + c] = v3d_c(0.0, 0.0, 1.0);

    for (unsigned int r = 0; r < rows; ++r)
        for (unsigned int c = cols - stripe_size; c < cols; ++c)
            grid_set_anisotropy_loc(&g, r, c, (anisotropy){.dir = v3d_c(0.0, 0.0, 1.0), .ani = 0.05 * J});


    v3d_create_skyrmion(g.m, g.gi.rows, g.gi.cols, 5, rows / 2.0, (cols - stripe_size) / 2, -1.0, -1.0, M_PI / 2.0);
    double dt = 0.01 * HBAR / (J * SIGN(J));

    string current_func = str_is_cstr("current ret = (current){};\n"\
                                      "time -= 0.5 * NS;\n"\
                                      "ret.type = CUR_STT;\n"\
                                      "ret.stt.j = v3d_c(2.05e10, 0.0, 0.0);\n"\
                                      "ret.stt.beta = 0.0;\n"\
                                      "ret.stt.polarization = -1.0;\n"\
                                      "double omega = 318194630.401;\n"\
                                      "double ac = 5.0e10 * sin(omega * time);\n"\
                                      "ret.stt.j = v3d_sum(ret.stt.j, v3d_c(0.0, ac, 0.0));\n"\
                                      "ret.stt.j = v3d_scalar(ret.stt.j, time > 0);\n"\
                                      "return ret;");

    string field_func = str_from_fmt("double Hz = %.15e;\n"\
                                     "return v3d_c(0.0, 0.0, Hz);", 0.5 * dm * dm / (J * SIGN(J)) * 1.0 / mu);

    string temperature_func = str_is_cstr("return 0.0;");
    string compile = str_is_cstr("-cl-fast-relaxed-math");

    srand(time(NULL));

    gpu_cl gpu = gpu_cl_init(current_func, field_func, temperature_func, STR_NULL, compile);
    str_free(&field_func);

    logging_log(LOG_INFO, "Integration dt: %e", dt);
    double ratio = (double)rows / cols;
#if 1
    integrate_context ctx = integrate_context_init(&g, &gpu, dt);
    grid_renderer_integration(&g, &gpu, ctx, 800, 800 * ratio);
#else
    integrate(&g, .dt=dt, .duration=20 * NS, .current_generation_function = current_func, .field_generation_function = field_func, .output_path = str_from_fmt("./out.dat"));
#endif
    grid_free(&g);
    return 0;
}
