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

    for(size_t I = 0; I < s.g_old.param.total; ++I)
        s.g_old.grid[I] = VecNormalize(field_joule);

    CreateSkyrmionNeel(s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols, 
                       s.g_old.param.cols / 4, s.g_old.param.rows / 2, 3, 1.0, 1.0);

    CreateSkyrmionNeel(s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols, 
                       3 * s.g_old.param.cols / 4, s.g_old.param.rows / 2, 3, 1.0, 1.0);

    for(size_t I = 0; I < s.g_old.param.total; ++I)
        GridNormalizeI(I, &s.g_old);

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
        IntegrateSimulator(&s, field_tesla, cur);
        s.doing_relax = false;
        printf("Done relaxing\n");
    }
        
    PrintVecGridToFile("./output/start.out", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols);
    DumpGrid("./output/start.bin", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols);
    printf("%d\n", s.doing_relax);
    if (s.use_gpu)
        WriteFullGridBuffer(s.gpu.queue, s.g_old_buffer, &s.g_old);

    J = 0.2e12;//20.0e9;
    J = RealCurToNorm(J, s.g_old.param);
    printf("%e\n", J);
    cur = (Current){VecFrom(0.0, -J, 0.0), -1.0, 0.0, 1.0e-9, CUR_STT};

    if (s.do_integrate)
        IntegrateSimulator(&s, field_tesla, cur);

    DumpGridCharge("./output/end_grid_charge.bin", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols, s.g_old.param.lattice, s.g_old.param.lattice, s.g_old.param.pbc);
    DumpWriteChargeGrid("./output/grid_charge_anim.bin", &s);
    DumpWriteGrid("./output/grid_anim_dump.bin", &s);
    DumpGrid("./output/end.bin", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols);
    PrintVecGridToFile("./output/end.out", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols);
    WriteSimulatorSimulation("./output/anim", &s, 0);
    FreeSimulator(&s);
    return 0;
}
