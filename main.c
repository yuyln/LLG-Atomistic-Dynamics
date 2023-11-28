#include "./headers/helpers.h"
#include "./headers/gsa.h"
#include "./headers/helpers_simulator.h"
#include "./headers/gradient_descent.h"

//TODO: Test regions
int main(int argc, const char **argv) {
    if (argc < 2) {
        printf("No input file provided\n");
        return 1;
    }

    simulator_t s = init_simulator(argv[1]);
    export_simulator_path(&s, "./output/export_sim.out");
    printf("Grid size in bytes: %zu\n", find_grid_size_bytes(&s.g_old));

    if (s.do_gsa) {
        dump_v3d_grid("./output/GSA_before.bin", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols, s.g_old.param.lattice);

        gsa(s.gsap, &s.g_old, &s.g_new, &s.gpu);
        grid_copy(&s.g_old, &s.g_new);
        full_grid_write_buffer(s.gpu.queue, s.g_old_buffer, &s.g_old);
        full_grid_write_buffer(s.gpu.queue, s.g_new_buffer, &s.g_new);

        dump_v3d_grid("./output/GSA_after.bin", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols, s.g_old.param.lattice);
    }


    if (s.use_gpu)
        full_grid_write_buffer(s.gpu.queue, s.g_old_buffer, &s.g_old);

    if (s.do_gradient) {
        dump_v3d_grid("./output/GRADIENT_before.bin", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols, s.g_old.param.lattice);

        gradient_descent(&s.g_old, &s.g_new, s.dt, s.alpha_gradient, s.beta_gradient, s.mass_gradient, s.gradient_steps, s.temp_gradient, s.factor_gradient, &s.gpu);
        copy_grid_to_allocated_grid(&s.g_old, &s.g_new);

        dump_v3d_grid("./output/GRADIENT_after.bin", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols, s.g_old.param.lattice);
    }

    if (s.use_gpu)
        full_grid_write_buffer(s.gpu.queue, s.g_old_buffer, &s.g_old);


    if (s.do_relax) {
        printf("Relaxing\n");
        s.doing_relax = true;
        dump_v3d_grid("./output/RELAX_before.bin", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols, s.g_old.param.lattice);
        integrate_simulator(&s, "./output/did_relax");
        dump_v3d_grid("./output/RELAX_after.bin", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols, s.g_old.param.lattice);
        s.doing_relax = false;
        printf("Done relaxing\n");
    }

    dump_v3d_grid("./output/INTEGRATION_before.bin", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols, s.g_old.param.lattice);
    if (s.use_gpu)
        full_grid_write_buffer(s.gpu.queue, s.g_old_buffer, &s.g_old);

    if (s.do_integrate)
        integrate_simulator(&s, "./output/integration_fly.bin");

    if (s.write_to_file)
        dump_write_grid("./output/grid_anim_dump.bin", &s);

    dump_v3d_grid("./output/end.bin", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols, s.g_old.param.lattice);
    print_v3d_grid_path("./output/end.out", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols);
    write_simulation_data("./output/anim", &s);
    free_simulator(&s);
    profiler_print_measures(stdout);
    return 0;
}
