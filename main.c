#include "./headers/helpers.h"
#include "./headers/gsa.h"
#include "./headers/helpers_simulator.h"
#include "./headers/gradient_descent.h"

//TODO: fix crashing when providing file that does not exist
int main() {
    double J_norm = 5.0e-2;
    double jx =  0.0,
           jy =  1.0,
           jz =  0.0;
    double p = -1.0;
    double beta = 0.0;
    CUR_TYPE cur_type = CUR_STT;
    double dh = 1.0e-9;

    double Hz_norm =  0.5,
           Hy_norm =  0.0,
           Hx_norm =  0.0;

    simulator_t s = init_simulator("./input/input.in");
    export_simulator_path(&s, "./output/export_sim.out");
    printf("Grid size in bytes: %zu\n", find_grid_size_bytes(&s.g_old));

    v3d field_joule = v3d_scalar(v3d_c(Hx_norm, Hy_norm, Hz_norm), s.g_old.param.dm * s.g_old.param.dm / s.g_old.param.exchange);
    v3d field_real = field_joule_to_tesla(v3d_scalar(field_joule, s.real_param.exchange), s.real_param.mu_s);
    current_t cur;

    if (s.do_gsa) {
        dump_v3d_grid("./output/GSA_before.bin", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols, s.g_old.param.lattice);

        if (s.use_gpu) {
            gsa_gpu(s.gsap, &s.g_old, &s.g_new, field_joule, &s.gpu);
            grid_copy(&s.g_old, &s.g_new);
            full_grid_write_buffer(s.gpu.queue, s.g_old_buffer, &s.g_old);
            full_grid_write_buffer(s.gpu.queue, s.g_new_buffer, &s.g_new);
        } else
            gsa(s.gsap, &s.g_old, &s.g_new, field_joule);

        dump_v3d_grid("./output/GSA_after.bin", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols, s.g_old.param.lattice);
    }


    if (s.use_gpu)
        full_grid_write_buffer(s.gpu.queue, s.g_old_buffer, &s.g_old);

    if (s.do_gradient) {
        dump_v3d_grid("./output/GRADIENT_before.bin", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols, s.g_old.param.lattice);

        gradient_descent(&s.g_old, &s.g_new, s.dt, s.alpha_gradient, s.beta_gradient, s.mass_gradient, s.gradient_steps, field_joule, s.temp_gradient, s.factor_gradient, &s.gpu, s.use_gpu, s.n_cpu);
        copy_grid_to_allocated_grid(&s.g_old, &s.g_new);

        dump_v3d_grid("./output/GRADIENT_after.bin", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols, s.g_old.param.lattice);
    }

    if (s.use_gpu)
        full_grid_write_buffer(s.gpu.queue, s.g_old_buffer, &s.g_old);


    if (s.do_relax) {
        cur = (current_t){0};
        printf("Relaxing\n");
        s.doing_relax = true;
        dump_v3d_grid("./output/RELAX_before.bin", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols, s.g_old.param.lattice);
        integrate_simulator(&s, field_joule, cur, "./output/did_relax");
        dump_v3d_grid("./output/RELAX_after.bin", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols, s.g_old.param.lattice);
        s.doing_relax = false;
        printf("Done relaxing\n");
    }

    dump_v3d_grid("./output/INTEGRATION_before.bin", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols, s.g_old.param.lattice);
    if (s.use_gpu)
        full_grid_write_buffer(s.gpu.queue, s.g_old_buffer, &s.g_old);

    printf("-------------------------------------\n");
    printf("Real values:\n");
    printf("current  || Normalized: %.5e Real: %.5e A/m^2\n", J_norm, current_normalized_to_real(J_norm, s.real_param));
    printf("Field    || Normalized: (%.5e, %.5e, %.5e) Real: (%.5e, %.5e, %.5e) T\n", field_joule.x, field_joule.y, field_joule.z,
                                                                                      field_real.x, field_real.y, field_real.z);
    printf("                       =(%.5e, %.5e, %.5e)D^2/J\n", Hx_norm, Hy_norm, Hz_norm);
    printf("-------------------------------------\n");
    v3d j = v3d_normalize(v3d_c(jx, jy, jz));
    cur = (current_t){.thick = dh, .theta_sh = 0.0, .j = J_norm, .P = p, .beta = beta, .type = cur_type,
                      .j_hat = j, .p_hat = j};

    if (s.do_integrate)
        integrate_simulator(&s, field_joule, cur, "./output/integration_fly.bin");

    if (s.write_to_file)
        dump_write_grid("./output/grid_anim_dump.bin", &s);

    dump_v3d_grid("./output/end.bin", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols, s.g_old.param.lattice);
    print_v3d_grid_path("./output/end.out", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols);
    write_simulation_data("./output/anim", &s);
    free_simulator(&s);
    return 0;
}
