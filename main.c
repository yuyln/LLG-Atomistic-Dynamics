#include "atomistic_simulation.h"
#include <time.h>
#include <math.h>

//@TODO: Change openclwrapper to print file and location correctly
//@TODO: Check uint64_t->int changes
//@TODO: Do 3D
//@TODO: This is a f*** mess, need to organize better later
//
int main(void) {
    render_window *window = window_init(800, 600);
    while (!window_should_close(window)) {
        window_render(window);
        window_poll(window);
    }
    /*srand(time(NULL));
    double dt = 6.0e-15;
    unsigned int rows = 64;
    unsigned int cols = 64;
    double ratio = (double)rows / cols;
    grid g_ = grid_init(rows, cols);
    grid *g = &g_;
    v3d_fill_with_random(g->m, g->gi.rows, g->gi.cols);
    grid_set_anisotropy(g, (anisotropy){.ani=0.02 * QE * 1.0e-3, .dir = v3d_c(0.0, 0.0, 1.0)});
    grid_set_dm(g, 0.6 * QE * 1.0e-3, 0.0, R_ij_CROSS_Z);
    //integrate(g, .output_path = sv_from_cstr("./simulation_info.csv"), .dt=dt);


    gpu_cl gpu = gpu_cl_init(0, 0);
    char *kernel;
    clw_read_file("./kernel_complete.cl", &kernel);
    const char cmp[] = "-DOPENCL_COMPILATION";
    string_view kernel_view = (string_view){.str = kernel, .len = strlen(kernel)};
    string_view compile_opt = (string_view){.str = (char *const)cmp, .len = strlen(cmp) + 1};
    gpu_cl_compile_source(&gpu, kernel_view, compile_opt);
    free(kernel);
    uint64_t step_id = gpu_append_kernel(&gpu, "gpu_step");
    uint64_t info_id = gpu_append_kernel(&gpu, "extract_info");
    uint64_t exchange_id = gpu_append_kernel(&gpu, "exchange_grid");
    uint64_t render_grid_id = gpu_append_kernel(&gpu, "render_grid_compa");
    uint64_t render_charge_id = gpu_append_kernel(&gpu, "render_topological_charge");
    uint64_t render_energy_id = gpu_append_kernel(&gpu, "render_energy");
    uint64_t render_magnetic_id = gpu_append_kernel(&gpu, "render_magnetic_field");
    uint64_t render_eletric_id = gpu_append_kernel(&gpu, "render_eletric_field");
    uint64_t render_id = render_grid_id;

    unsigned int w_width = 800;
    unsigned int w_height = w_width * ratio;

    grid_to_gpu(g, gpu);

    cl_char4 *rgba = calloc(w_width * w_height, sizeof(cl_char4));

    cl_mem swap_buffer = clw_create_buffer(g->gi.rows * g->gi.cols * sizeof(*g->m), gpu.ctx, CL_MEM_READ_WRITE);
    cl_mem rgba_buffer = clw_create_buffer(w_width * w_height * sizeof(cl_char4), gpu.ctx, CL_MEM_READ_WRITE);

    information_packed *info = calloc(g->gi.rows * g->gi.cols, sizeof(information_packed));
    cl_mem info_buffer = clw_create_buffer(g->gi.rows * g->gi.cols * sizeof(*info), gpu.ctx, CL_MEM_READ_WRITE);

    int quit = 0;
    double passed = 0.0;
    int factor = 100;
    uint64_t step = 0;

    clw_set_kernel_arg(gpu.kernels[step_id], 0, sizeof(cl_mem), &g->gp_buffer);
    clw_set_kernel_arg(gpu.kernels[step_id], 1, sizeof(cl_mem), &g->m_buffer);
    clw_set_kernel_arg(gpu.kernels[step_id], 2, sizeof(cl_mem), &swap_buffer);
    clw_set_kernel_arg(gpu.kernels[step_id], 3, sizeof(double), &dt);
    clw_set_kernel_arg(gpu.kernels[step_id], 4, sizeof(double), &passed);
    clw_set_kernel_arg(gpu.kernels[step_id], 5, sizeof(grid_info), &g->gi);

    clw_set_kernel_arg(gpu.kernels[info_id], 0, sizeof(cl_mem), &g->gp_buffer);
    clw_set_kernel_arg(gpu.kernels[info_id], 1, sizeof(cl_mem), &g->m_buffer);
    clw_set_kernel_arg(gpu.kernels[info_id], 2, sizeof(cl_mem), &swap_buffer);
    clw_set_kernel_arg(gpu.kernels[info_id], 3, sizeof(cl_mem), &info_buffer);
    clw_set_kernel_arg(gpu.kernels[info_id], 4, sizeof(double), &dt);
    clw_set_kernel_arg(gpu.kernels[info_id], 5, sizeof(double), &passed);
    clw_set_kernel_arg(gpu.kernels[info_id], 6, sizeof(grid_info), &g->gi);

    clw_set_kernel_arg(gpu.kernels[exchange_id], 0, sizeof(cl_mem), &g->m_buffer);
    clw_set_kernel_arg(gpu.kernels[exchange_id], 1, sizeof(cl_mem), &swap_buffer);


    clw_set_kernel_arg(gpu.kernels[render_id], 0, sizeof(cl_mem), &info_buffer);
    clw_set_kernel_arg(gpu.kernels[render_id], 1, sizeof(grid_info), &g->gi);
    clw_set_kernel_arg(gpu.kernels[render_id], 2, sizeof(double), &dt); //useless
    clw_set_kernel_arg(gpu.kernels[render_id], 3, sizeof(double), &dt); //useless
    clw_set_kernel_arg(gpu.kernels[render_id], 4, sizeof(cl_mem), &rgba_buffer);
    clw_set_kernel_arg(gpu.kernels[render_id], 5, sizeof(unsigned int), &w_width);
    clw_set_kernel_arg(gpu.kernels[render_id], 6, sizeof(unsigned int), &w_height);

    size_t global_sim = g->gi.rows * g->gi.cols;
    size_t local_sim = clw_gcd(global_sim, 32);

    size_t global_ren = w_width * w_height;
    size_t local_ren = clw_gcd(global_ren, 32);



    FILE *output_info = fopen("./simulation_info.csv", "w");
    fprintf(output_info, "time(s),energy(eV),exchange_energy(eV),dm_energy(eV),field_energy(eV),anisotropy_energy(eV),cubic_anisotropy_energy(eV),");
    fprintf(output_info, "charge_finite,charge_lattice,");
    fprintf(output_info, "avg_mx,avg_my,avg_mz,");
    fprintf(output_info, "eletric_x,eletric_y,eletric_z,");
    fprintf(output_info, "magnetic_lattice_x,magnetic_lattice_y,magnetic_lattice_z,");
    fprintf(output_info, "magnetic_derivative_x,magnetic_derivative_y,magnetic_derivative_z\n");

    while (!quit) {
        double charge_min = FLT_MAX;
        double charge_max = -FLT_MAX;

        double magnetic_min = FLT_MAX;
        double magnetic_max = -FLT_MAX;

        double eletric_min = FLT_MAX;
        double eletric_max = -FLT_MAX;

        double energy_min = FLT_MAX;
        double energy_max = -FLT_MAX;

        for (int A = 0; A < factor; ++A) {
            integrate_step(passed, &gpu, step_id, global_sim, local_sim);

            if (step % 100 == 0) {
                integrate_get_info(passed, &gpu, info_id, global_sim, local_sim);
                clw_print_cl_error(stderr, clEnqueueReadBuffer(gpu.queue, info_buffer, CL_TRUE, 0, sizeof(*info) * g->gi.rows * g->gi.cols, info, 0, NULL, NULL), "[ FATAL ] Could not read info_buiffer");
                information_packed local = {0};
                for (uint64_t i = 0; i < g->gi.rows * g->gi.cols; ++i) {
                    local.energy += info[i].energy;
                    local.cubic_energy += info[i].cubic_energy;
                    local.anisotropy_energy += info[i].anisotropy_energy;
                    local.field_energy += info[i].field_energy;
                    local.dm_energy += info[i].dm_energy;
                    local.exchange_energy += info[i].exchange_energy;
                    local.charge_finite += info[i].charge_finite;
                    local.charge_lattice += info[i].charge_lattice;
                    local.avg_m = v3d_sum(local.avg_m, v3d_scalar(info[i].avg_m, 1.0 / (g->gi.rows * g->gi.cols)));
                    local.eletric_field = v3d_sum(local.eletric_field , info[i].eletric_field);
                    local.magnetic_field_lattice = v3d_sum(local.magnetic_field_lattice, info[i].magnetic_field_lattice);
                    local.magnetic_field_derivative = v3d_sum(local.magnetic_field_derivative, info[i].magnetic_field_derivative);

                    charge_min = charge_min < info[i].charge_lattice? charge_min: info[i].charge_lattice;

                    magnetic_min = magnetic_min < info[i].magnetic_field_lattice.x? magnetic_min: info[i].magnetic_field_lattice.x;
                    magnetic_min = magnetic_min < info[i].magnetic_field_lattice.y? magnetic_min: info[i].magnetic_field_lattice.y;
                    magnetic_min = magnetic_min < info[i].magnetic_field_lattice.z? magnetic_min: info[i].magnetic_field_lattice.z;

                    eletric_min = eletric_min < info[i].eletric_field.x? eletric_min: info[i].eletric_field.x;
                    eletric_min = eletric_min < info[i].eletric_field.y? eletric_min: info[i].eletric_field.y;
                    eletric_min = eletric_min < info[i].eletric_field.z? eletric_min: info[i].eletric_field.z;

                    energy_min = energy_min < info[i].energy? energy_min: info[i].energy;

                    charge_max = charge_max > info[i].charge_lattice? charge_max: info[i].charge_lattice;

                    magnetic_max = magnetic_max > info[i].magnetic_field_lattice.x? magnetic_max: info[i].magnetic_field_lattice.x;
                    magnetic_max = magnetic_max > info[i].magnetic_field_lattice.y? magnetic_max: info[i].magnetic_field_lattice.y;
                    magnetic_max = magnetic_max > info[i].magnetic_field_lattice.z? magnetic_max: info[i].magnetic_field_lattice.z;

                    eletric_max = eletric_max > info[i].eletric_field.x? eletric_max: info[i].eletric_field.x;
                    eletric_max = eletric_max > info[i].eletric_field.y? eletric_max: info[i].eletric_field.y;
                    eletric_max = eletric_max > info[i].eletric_field.z? eletric_max: info[i].eletric_field.z;

                    energy_max = energy_max > info[i].energy? energy_max: info[i].energy;
                }
                fprintf(output_info, "%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,", passed, local.energy, local.exchange_energy, local.dm_energy, local.field_energy, local.anisotropy_energy, local.cubic_energy);
                fprintf(output_info, "%.15e,%.15e,", local.charge_finite, local.charge_lattice);
                fprintf(output_info, "%.15e,%.15e,%.15e,", local.avg_m.x, local.avg_m.y, local.avg_m.z);
                fprintf(output_info, "%.15e,%.15e,%.15e,", local.eletric_field.x, local.eletric_field.y, local.eletric_field.z);
                fprintf(output_info, "%.15e,%.15e,%.15e,", local.magnetic_field_lattice.x, local.magnetic_field_lattice.y, local.magnetic_field_lattice.z);
                fprintf(output_info, "%.15e,%.15e,%.15e\n", local.magnetic_field_derivative.x, local.magnetic_field_derivative.y, local.magnetic_field_derivative.z);
            }

            integrate_exchange_grids(&gpu, exchange_id, global_sim, local_sim);

            passed += dt;
            step++;
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
                case KeyPress: {
                    switch (XLookupKeysym(&event.xkey, 0)) {
                        case 'q': {
                            render_id = render_charge_id;
                            clw_set_kernel_arg(gpu.kernels[render_id], 2, sizeof(double), &charge_min);
                            clw_set_kernel_arg(gpu.kernels[render_id], 3, sizeof(double), &charge_max);
                        }
                            break;
                        case 'g': {
                            render_id = render_grid_id;
                            clw_set_kernel_arg(gpu.kernels[render_id], 2, sizeof(double), &charge_min);
                            clw_set_kernel_arg(gpu.kernels[render_id], 3, sizeof(double), &charge_max);
                        }
                            break;
                        case 'e': {
                            render_id = render_energy_id;
                            clw_set_kernel_arg(gpu.kernels[render_id], 2, sizeof(double), &energy_min);
                            clw_set_kernel_arg(gpu.kernels[render_id], 3, sizeof(double), &energy_max);
                        }
                            break;
                        case 'b': {
                            render_id = render_magnetic_id;
                            clw_set_kernel_arg(gpu.kernels[render_id], 2, sizeof(double), &magnetic_min);
                            clw_set_kernel_arg(gpu.kernels[render_id], 3, sizeof(double), &magnetic_max);
                        }
                            break;
                        case 't': {
                            render_id = render_eletric_id;
                            clw_set_kernel_arg(gpu.kernels[render_id], 2, sizeof(double), &eletric_min);
                            clw_set_kernel_arg(gpu.kernels[render_id], 3, sizeof(double), &eletric_max);
                        }
                            break;
                        default:
                            break;
                    }
                    clw_set_kernel_arg(gpu.kernels[render_id], 0, sizeof(cl_mem), &info_buffer);
                    clw_set_kernel_arg(gpu.kernels[render_id], 1, sizeof(grid_info), &g->gi);
                    clw_set_kernel_arg(gpu.kernels[render_id], 4, sizeof(cl_mem), &rgba_buffer);
                    clw_set_kernel_arg(gpu.kernels[render_id], 5, sizeof(unsigned int), &w_width);
                    clw_set_kernel_arg(gpu.kernels[render_id], 6, sizeof(unsigned int), &w_height);
                }
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
    clw_print_cl_error(stderr, clReleaseMemObject(info_buffer), "[ FATAL ] Could not release info buffer from GPU");

    grid_release_from_gpu(g);
    gpu_cl_close(&gpu);
    free(rgba);

    grid_free(&g_);
    fclose(output_info);*/
    return 0;
}
