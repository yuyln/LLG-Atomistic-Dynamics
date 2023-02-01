#include <helpers.h>
#include <gsa.h>
#include <helpers_simulator.h>

int main()
{
    Simulator s = InitSimulator("./input/input.in");
    ExportSimulatorFile(&s, "./output/export_sim.out");
    printf("Grid size in bytes: %zu\n", FindGridSize(&s.g_old));
    PrintVecGridToFile("./output/before.out", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols);

    
    Vec field_joule = VecFrom(0.0, 0.0, -0.5 * s.g_old.param.dm * s.g_old.param.dm / s.g_old.param.exchange);
    Vec field_tesla = FieldJouleToTesla(field_joule, s.g_old.param.mu_s);
    for (size_t i = 0; i < s.g_old.param.total; ++i)
        s.g_old.grid[i] = VecNormalize(field_joule);

    CreateSkyrmionBloch(s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols,
                        s.g_old.param.cols / 2, s.g_old.param.rows / 2,
                        6, 1, 1);

    double J; Current cur;

    if (s.use_gpu && s.do_gsa)
    {
        GSAGPU(s.gsap, &s.g_old, &s.g_new, field_tesla, &s.gpu);
        CopyGrid(&s.g_old, &s.g_new);
        WriteFullGridBuffer(s.gpu.queue, s.g_old_buffer, &s.g_old);
        WriteFullGridBuffer(s.gpu.queue, s.g_new_buffer, &s.g_new);
    }
    else if (s.do_gsa)
    {
        GSA(s.gsap, &s.g_old, &s.g_new, field_tesla);
    }

    if (s.use_gpu)
    {
        WriteFullGridBuffer(s.gpu.queue, s.g_old_buffer, &s.g_old);
    }

    if (s.do_relax)
    {
        J = 0.0;
        cur = (Current){0};
        printf("Relaxing\n");
        s.doing_relax = true;
        IntegrateSimulator(&s, field_tesla, cur, "./output/did_relax");
        s.doing_relax = false;
        printf("Done relaxing\n");
    }
        
    PrintVecGridToFile("./output/start.out", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols);
    DumpGrid("./output/start.bin", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols);
    printf("%d\n", s.doing_relax);
    if (s.use_gpu)
        WriteFullGridBuffer(s.gpu.queue, s.g_old_buffer, &s.g_old);

    J = 1.234147e-02;
    // J = RealCurToNorm(0.150e12, s.g_old.param);
    printf("Norm: %e Real: %e\n", J, NormCurToReal(J, s.g_old.param));
    cur = (Current){VecFrom(0.0, -J, 0.0), -1.0, 0.0, 1.0e-9, CUR_STT};

    if (s.do_integrate)
        IntegrateSimulator(&s, field_tesla, cur, "./output/integration_fly.bin");

    if (s.write_to_file)
    {
        // DumpGridCharge("./output/end_grid_charge.bin", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols, s.g_old.param.lattice, s.g_old.param.lattice, s.g_old.param.pbc);
        // DumpWriteChargeGrid("./output/grid_charge_anim.bin", &s);
        DumpWriteGrid("./output/grid_anim_dump.bin", &s);
    }

    DumpGrid("./output/end.bin", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols);
    PrintVecGridToFile("./output/end.out", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols);
    WriteSimulatorSimulation("./output/anim", &s);
    FreeSimulator(&s);
    return 0;
}
