#include <helpers.h>
#include <gsa.h>
#include <helpers_simulator.h>

int main()
{
    Simulator s = InitSimulator("./input/input.in");
    ExportSimulatorFile(&s, "./output/export_sim.out");
    printf("Grid size in bytes: %zu\n", FindGridSize(&s.g_old));
    PrintVecGridToFile("./output/before.out", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols);

    
    Vec field_joule = VecFrom(0.0, 0.0, 0.5 * s.g_old.param.dm * s.g_old.param.dm / s.g_old.param.exchange);
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

    for(int col = 0; col < s.g_old.param.cols; ++col)
    {
        s.g_old.grid[col] = VecNormalize(field_joule);
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
        IntegrateSimulator(&s, field_tesla, cur);
        s.doing_relax = false;
        printf("Done relaxing\n");
    }
        
    PrintVecGridToFile("./output/start.out", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols);
    printf("%d\n", s.doing_relax);
    if (s.use_gpu)
        WriteFullGridBuffer(s.gpu.queue, s.g_old_buffer, &s.g_old);

    J = 0.2e12;//20.0e9;
    J = RealCurToNorm(J, s.g_old.param);
    printf("%e\n", J);
    cur = (Current){VecFrom(0.0, -J, 0.0), -1.0, 0.0, 1.0e-9, CUR_STT};

    if (s.do_integrate)
        IntegrateSimulator(&s, field_tesla, cur);

    DumpWriteGrid("./output/grid_dump.bin", &s);
    PrintVecGridToFile("./output/end.out", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols);
    WriteSimulatorSimulation("./output/anim", &s);
    FreeSimulator(&s);
    return 0;
}
