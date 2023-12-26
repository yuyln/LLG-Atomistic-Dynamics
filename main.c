#include <time.h>
#include <stdlib.h>

#include <X11/Xlib.h>
#define XK_LATIN1
#include <X11/keysymdef.h>
#include <X11/extensions/Xdbe.h>

#include "grid_funcs.h"
#include "integrate.h"
#include "simulation_funcs.h"

//@TODO: Change openclwrapper to print file and location correctly
//@TODO: Check uint64_t->int changes
//@TODO: Do 3D
//@TODO: This is a f*** mess, need to organize better later
int main(void) {
    srand(time(NULL));
    double dt = 6.0e-15;
    unsigned int rows = 64;
    unsigned int cols = 64;
    double ratio = (double)rows / cols;
    grid g_ = grid_init(rows, cols);
    grid *g = &g_;
    v3d_fill_with_random(g->m, g->gi.rows, g->gi.cols);
    grid_set_anisotropy(g, (anisotropy){.ani=0.02 * QE * 1.0e-3, .dir = v3d_c(0.0, 0.0, 1.0)});
    grid_set_dm(g, 1.0 * QE * 1.0e-3, 0.0, R_ij);
    g->gi.pbc.dirs = 0;
    integrate(g);
    return 0;
    //integrate(&g, dt, 1.0 * NS, 100, 1000, (string_view){0}, (string_view){0}, "./output/");


    gpu_cl gpu = gpu_cl_init(0, 0);
    char *kernel;
    clw_read_file("./kernel_complete.cl", &kernel);
    const char cmp[] = "-DOPENCL_COMPILATION";
    string_view kernel_view = (string_view){.str = kernel, .len = strlen(kernel)};
    string_view compile_opt = (string_view){.str = (char *const)cmp, .len = strlen(cmp) + 1};
    gpu_cl_compile_source(&gpu, kernel_view, compile_opt);
    free(kernel);
    uint64_t step_id = gpu_append_kernel(&gpu, "gpu_step");
    uint64_t exchange_id = gpu_append_kernel(&gpu, "exchange_grid");
    uint64_t to_rgb_id = gpu_append_kernel(&gpu, "v3d_to_rgb");
    uint64_t render_id = gpu_append_kernel(&gpu, "render_grid");

    unsigned int w_width = 800;
    unsigned int w_height = w_width * ratio;

    grid_to_gpu(g, gpu);

    cl_char4 *rgba = calloc(w_width * w_height, sizeof(cl_char4));

    cl_mem swap_buffer = clw_create_buffer(g->gi.rows * g->gi.cols * sizeof(*g->m), gpu.ctx, CL_MEM_READ_WRITE);
    cl_mem rgba_buffer = clw_create_buffer(w_width * w_height * sizeof(cl_char4), gpu.ctx, CL_MEM_READ_WRITE);

    clw_set_kernel_arg(gpu.kernels[step_id], 0, sizeof(cl_mem), &g->gp_buffer);
    clw_set_kernel_arg(gpu.kernels[step_id], 1, sizeof(cl_mem), &g->m_buffer);
    clw_set_kernel_arg(gpu.kernels[step_id], 2, sizeof(cl_mem), &swap_buffer);
    clw_set_kernel_arg(gpu.kernels[step_id], 3, sizeof(double), &dt);
    clw_set_kernel_arg(gpu.kernels[step_id], 5, sizeof(grid_info), &g->gi);

    clw_set_kernel_arg(gpu.kernels[exchange_id], 0, sizeof(cl_mem), &g->m_buffer);
    clw_set_kernel_arg(gpu.kernels[exchange_id], 1, sizeof(cl_mem), &swap_buffer);

    size_t global_sim = g->gi.rows * g->gi.cols;
    size_t local_sim = clw_gcd(global_sim, 32);

    size_t global_ren = w_width * w_height;
    size_t local_ren = clw_gcd(global_ren, 32);

    Display *display = XOpenDisplay(NULL);
    if (display == NULL) {
        fprintf(stderr, "ERROR: could not open the default display\n");
        exit(1);
    }

    int major_version_return, minor_version_return;
    if(XdbeQueryExtension(display, &major_version_return, &minor_version_return)) {
        printf("XDBE version %d.%d\n", major_version_return, minor_version_return);
    } else {
        fprintf(stderr, "XDBE is not supported!!!1\n");
        exit(1);
    }


    clw_set_kernel_arg(gpu.kernels[render_id], 0, sizeof(cl_mem), &g->m_buffer);
    clw_set_kernel_arg(gpu.kernels[render_id], 1, sizeof(unsigned int), &g->gi.rows);
    clw_set_kernel_arg(gpu.kernels[render_id], 2, sizeof(unsigned int), &g->gi.cols);
    clw_set_kernel_arg(gpu.kernels[render_id], 3, sizeof(cl_mem), &rgba_buffer);
    clw_set_kernel_arg(gpu.kernels[render_id], 4, sizeof(unsigned int), &w_width);
    clw_set_kernel_arg(gpu.kernels[render_id], 5, sizeof(unsigned int), &w_height);

    Window window = XCreateSimpleWindow(
            display,
            XDefaultRootWindow(display),
            0, 0,
            w_width, w_height,
            0,
            0,
            0);

    XdbeBackBuffer back_buffer = XdbeAllocateBackBufferName(display, window, 0);
    printf("back_buffer ID: %lu\n", back_buffer);

    XWindowAttributes wa = {0};
    XGetWindowAttributes(display, window, &wa);

    XImage *image = XCreateImage(display,
            wa.visual,
            wa.depth,
            ZPixmap,
            0,
            (char*) rgba,
            w_width,
            w_height,
            sizeof(cl_char4) * 8,
            w_width * sizeof(cl_char4));

    GC gc = XCreateGC(display, window, 0, NULL);

    Atom wm_delete_window = XInternAtom(display, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(display, window, &wm_delete_window, 1);

    XSelectInput(display, window, KeyPressMask);

    XMapWindow(display, window);

    int quit = 0;
    double passed = 0.0;
    int factor = 5;
    while (!quit) {
        for (int A = 0; A < factor; ++A) {
            integrate_step(passed, &gpu, step_id, exchange_id, global_sim, local_sim);
            passed += dt;
        }
        clw_enqueue_nd(gpu.queue, gpu.kernels[render_id], 1, NULL, &global_ren, &local_ren);
        clw_read_buffer(rgba_buffer, rgba, sizeof(cl_char4) * w_width * w_height, 0, gpu.queue);
        //printf("%e\n", generate_magnetic_field(g->gp[0], passed).z);

        while (XPending(display) > 0) {
            XEvent event = {0};
            XNextEvent(display, &event);
            switch (event.type) {
                case ClientMessage: {
                    if ((Atom) event.xclient.data.l[0] == wm_delete_window) {
                        quit = 1;
                    }
                }
                break;
                default:
                    break;
            }
        }


        XPutImage(display, back_buffer, gc, image,
                0, 0,
                0, 0,
                w_width,
                w_height);
        XdbeSwapInfo swap_info;
        swap_info.swap_window = window;
        swap_info.swap_action = 0;
        XdbeSwapBuffers(display, &swap_info, 1);
    }

    XCloseDisplay(display);


    clw_print_cl_error(stderr, clReleaseMemObject(swap_buffer), "[ FATAL ] Could not release swap buffer from GPU");
    clw_print_cl_error(stderr, clReleaseMemObject(rgba_buffer), "[ FATAL ] Could not release rgb buffer from GPU");

    grid_release_from_gpu(g);
    gpu_cl_close(&gpu);
    free(rgba);

    grid_free(&g_);
    return 0;
}
