#include <helpers.h>
#include <gsa.h>
#include <helpers_simulator.h>
#include <gradient_descent.h>

int main()
{
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

    Simulator s = InitSimulator("./input/input.in");
    ExportSimulatorFile(&s, "./output/export_sim.out");
    printf("Grid size in bytes: %zu\n", FindGridSize(&s.g_old));

    Vec field_joule = VecScalar(VecFrom(Hx_norm, Hy_norm, Hz_norm), s.g_old.param.dm * s.g_old.param.dm / s.g_old.param.exchange);
    Vec field_tesla = FieldJouleToTesla(field_joule, s.g_old.param.mu_s);


    for (size_t i = 0; i < s.g_old.param.total; ++i)
        s.g_old.grid[i] = VecNormalize(field_tesla);
    CreateSkyrmionNeel(s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols, s.g_old.param.cols, s.g_old.param.cols - 12, s.g_old.param.rows - 3, 3, 1.0, -1.0);

    Current cur;

    if (s.use_gpu && s.do_gsa)
    {
        GSAGPU(s.gsap, &s.g_old, &s.g_new, field_tesla, &s.gpu);
        CopyGrid(&s.g_old, &s.g_new);
        WriteFullGridBuffer(s.gpu.queue, s.g_old_buffer, &s.g_old);
        WriteFullGridBuffer(s.gpu.queue, s.g_new_buffer, &s.g_new);
    }
    else if (s.do_gsa)
        GSA(s.gsap, &s.g_old, &s.g_new, field_tesla);

    if (s.use_gpu)
        WriteFullGridBuffer(s.gpu.queue, s.g_old_buffer, &s.g_old);

    if (s.do_gradient)
    {
        GradientDescent(&s.g_old, &s.g_new, s.dt, s.alpha_gradient, s.beta_gradient, s.mass_gradient, s.gradient_steps, field_tesla, s.temp_gradient, s.factor_gradient, &s.gpu, s.use_gpu, s.n_cpu);
        CopyToExistingGrid(&s.g_old, &s.g_new);
    }

    if (s.use_gpu)
        WriteFullGridBuffer(s.gpu.queue, s.g_old_buffer, &s.g_old);


    if (s.do_relax)
    {
        cur = (Current){0};
        printf("Relaxing\n");
        s.doing_relax = true;
        IntegrateSimulator(&s, field_tesla, cur, "./output/did_relax");
        s.doing_relax = false;
        printf("Done relaxing\n");
    }

    DumpGrid("./output/start.bin", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols, s.g_old.param.lattice);
    PrintVecGridToFile("./output/before_integration.out", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols);
    if (s.use_gpu)
        WriteFullGridBuffer(s.gpu.queue, s.g_old_buffer, &s.g_old);

    printf("-------------------------------------\n");
    printf("Real values:\n");
    printf("Current  || Normalized: %.5e Real: %.5e A/m^2\n", J_norm, NormCurToReal(J_norm, s.g_old.param));
    printf("Field    || Normalized: (%.5e, %.5e, %.5e) Real: (%.5e, %.5e, %.5e) T\n", field_joule.x, field_joule.y, field_joule.z,
                                                                                      field_tesla.x, field_tesla.y, field_tesla.z);
    printf("                       =(%.5e, %.5e, %.5e)D^2/J\n", Hx_norm, Hy_norm, Hz_norm);
    printf("-------------------------------------\n");
    Vec j = VecNormalize(VecFrom(jx, jy, jz));
    cur = (Current){VecScalar(j, J_norm), p, beta, dh, cur_type};

    if (s.do_integrate)
        IntegrateSimulator(&s, field_tesla, cur, "./output/integration_fly.bin");

    if (s.write_to_file)
        DumpWriteGrid("./output/grid_anim_dump.bin", &s);

    DumpGrid("./output/end.bin", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols, s.g_old.param.lattice);
    PrintVecGridToFile("./output/end.out", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols);
    WriteSimulatorSimulation("./output/anim", &s);
    FreeSimulator(&s);
    return 0;
}
