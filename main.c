#include "atomistic_simulation.h"

//@TODO: Change openclwrapper to print file and location correctly
//@TODO: Check uint64_t->int changes
//@TODO: Do 3D
//@TODO: This is a f*** mess, need to organize better later
//@TODO: Clear integrate ctx
int main(void) {
    double dt = 5.0e-15;
    int rows = 272;
    int cols = 272;
    double ratio = (double)cols / rows;
    render_window *window = window_init(800, 800 / ratio);
    grid g = grid_init(rows, cols);
    g.gi.pbc.m = v3d_normalize(v3d_c(1.0, 0.0, 0.0));
    for (int i = 0; i < rows * cols; ++i)
        g.m[i] = v3d_c(0.0, 0.0, 1.0);
    v3d_create_skyrmion(g.m, g.gi.rows, g.gi.cols, 30, rows / 2.0, cols / 2.0, -1.0, 1.0, M_PI / 2.0);
        //g.m[i] = v3d_normalize(v3d_c(shit_random(-1.0, 1.0), shit_random(-1.0, 1.0), shit_random(-1.0, 1.0)));

    grid_set_dm(&g, 0.1 * QE * 1.0e-3, 0.0, R_ij_CROSS_Z);
    //grid_set_anisotropy(&g, (anisotropy){.ani = 0.01 * QE * 1.0e-3, .dir = v3d_c(0.0, 0.0, 1.0)});

    //for (int row = 0; row < g.gi.rows; ++row) {
    //    for (int col = 0; col < g.gi.cols / 2; ++col) {
    //        grid_set_anisotropy_loc(&g, row, col, (anisotropy){.ani = -2.0 * QE * 1.0e-3, .dir = v3d_c(0.0, 0.0, 1.0)});
    //    }
    //}

    
    //integrate(&g, .dt=dt);

    grid_renderer gr = grid_renderer_init(&g, window, sv_from_cstr("current ret = (current){0};\nret.type = CUR_STT;\nret.stt.j = v3d_c((time > 0.1 * NS) * 1.0e11, 0.0, 0.0);\nret.stt.polarization = -1.0;\nret.stt.beta = 0.0;\nreturn ret;"),
                                                      sv_from_cstr("double normalized = 0.5;\ndouble real = normalized * gs.dm * gs.dm / gs.exchange / gs.mu;\nreturn v3d_c(0.0, 0.0, real);"),
                                                      (string_view){0},
                                                      (string_view){0});
    integrate_context ctx = integrate_context_init(&g, &gr.gpu, dt);

    int state = 'h';
    while(!window_should_close(window)) {
        switch (state) {
            case 'q':
                grid_renderer_charge(&gr);
                break;
            case 'e':
                grid_renderer_energy(&gr, ctx.time);
                break;
            case 'h':
                grid_renderer_hsl(&gr);
                break;
            case 'b':
                grid_renderer_bwr(&gr);
                break;
            default:
                grid_renderer_hsl(&gr);
        }
        if (window_key_pressed(window, 'q'))
            state = 'q';
        else if (window_key_pressed(window, 'e'))
            state = 'e';
        else if (window_key_pressed(window, 'h'))
            state = 'h';
        else if (window_key_pressed(window, 'b'))
            state = 'b';

        for (int i = 0; i < 100; ++i) {
            integrate_step(&ctx);
            integrate_exchange_grids(&ctx);
            ctx.time += dt;
        }

        window_render(window);
        window_poll(window);
    }

    integrate_context_close(&ctx);
    grid_renderer_close(&gr);
    grid_free(&g);
    window_close(window);
    return 0;
}
