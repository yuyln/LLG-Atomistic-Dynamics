#include "atomistic_simulation.h"

//@TODO: Change openclwrapper to print file and location correctly
//@TODO: Check uint64_t->int changes
//@TODO: Do 3D
//@TODO: This is a f*** mess, need to organize better later
//@TODO: Clear integrate ctx
int main(void) {
    double dt = 5.0e-15;
    render_window *window = window_init(800, 800);
    grid g = grid_init(64, 64);
    for (int i = 0; i < 64 * 64; ++i)
        g.m[i] = v3d_normalize(v3d_c(shit_random(-1.0, 1.0), shit_random(-1.0, 1.0), shit_random(-1.0, 1.0)));
    grid_set_dm(&g, 0.6 * QE * 1.0e-3, 0.0, R_ij_CROSS_Z);
    grid_renderer gr = grid_renderer_init(&g, window, (string_view){0}, (string_view){0}, (string_view){0}, (string_view){0});
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
     return 0;
}
