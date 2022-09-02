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

    double J = 0.0;
    J = RealCurToNorm(J, s.g_old.param);
    Current cur = (Current){VecFrom(J, 0.0, 0.0), -1.0, 0.0, 1.0e-9, CUR_NONE};

    /*GSAParam gsap = {2.8, 2.2, 2.6, 2.0, 70000, 10, 1};
    if (s.use_gpu)
    {
        GSAGPU(gsap, &s.g_old, &s.g_new, field_tesla, &s.gpu);
        CopyGrid(&s.g_old, &s.g_new);
        WriteFullGridBuffer(s.gpu.queue, s.g_old_buffer, &s.g_old);
        WriteFullGridBuffer(s.gpu.queue, s.g_new_buffer, &s.g_new);
    }
    else
    {
        GSA(gsap, &s.g_old, &s.g_new, field_tesla);
    }
*/
    //IntegrateSimulator(&s, field_tesla, cur);

    for (size_t I = 0; I < s.g_old.param.total; ++I)
        s.g_old.grid[I] = VecFrom(0.0, 0.0, -1.0);
    
    CreateSkyrmionNeel(s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols, s.g_old.param.cols / 3, s.g_old.param.rows / 2, 6, 1.0, 1.0);

    for (size_t I = 0; I < s.g_old.param.total; ++I)
        GridNormalizeI(I, &s.g_old);

    for (size_t I = 0; I < s.g_old.param.total; ++I)
    {
        int j = I % s.g_old.param.cols;
        int i = (I - j) / s.g_old.param.cols;
        s.g_old.grid[I] = VecScalar(s.g_old.grid[I], pow(1.0, i + j));
    }
    
    PrintVecGridToFile("./output/start.out", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols);

    if (s.use_gpu)
        WriteFullGridBuffer(s.gpu.queue, s.g_old_buffer, &s.g_old);

    J = 0.3e12;//20.0e9;
    J = RealCurToNorm(J, s.g_old.param);
    printf("%e\n", J);
    // cur = (Current){VecFrom(J, 0.0, 0.0), -1.0, 0.0, 1.0e-9, CUR_STT};
    cur = (Current){VecFrom(0.0, J, 0.0), 1.0, 0.0, 1.0e-9, CUR_CPP};

    IntegrateSimulator(&s, field_tesla, cur);

    PrintVecGridToFile("./output/end.out", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols);
    WriteSimulatorSimulation("./output/anim", &s);

    FreeSimulator(&s);
    return 0;
}

int main3()
{
    Simulator s = InitSimulator("./input/input.in");
    ExportSimulatorFile(&s, "./output/export_sim.out");

    Vec field_joule = VecFrom(0.0, 0.0, 0.5 * s.g_old.param.dm * s.g_old.param.dm / s.g_old.param.exchange);
    Vec field_tesla = FieldJouleToTesla(field_joule, s.g_old.param.mu_s);
    GSAParam gsap = {2.8, 2.2, 2.6, 2.0, 70000, 10, 1};
    printf("Before GSA: %e\n", Hamiltonian(&s.g_old, field_tesla));
    if (s.use_gpu)
        GSAGPU(gsap, &s.g_old, &s.g_new, field_tesla, &s.gpu);
    else
        GSA(gsap, &s.g_old, &s.g_new, field_tesla);

    CopyGrid(&s.g_old, &s.g_new);
    printf("After GSA: %e\n", Hamiltonian(&s.g_old, field_tesla));

    PrintVecGridToFile("./output/new.vec", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols);
    
    FreeSimulator(&s);
    return 0;
}

int main1()
{
    Simulator s = InitSimulator("./input/input.in");

    Vec field_joule = VecFrom(0.0, 0.0, 0.5 * s.g_old.param.dm * s.g_old.param.dm / s.g_old.param.exchange);
    Vec field_tesla = FieldJouleToTesla(field_joule, s.g_old.param.mu_s);

    printf("%e\n", field_tesla.z);
    PrintVecGridToFile("./output/old.vec", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols);
    
    printf("Before GSA %e J\n", Hamiltonian(&s.g_old, field_tesla));

    GSAParam gsap = {2.8, 2.2, 2.6, 2.0, 7, 1, 1};

    GSA(gsap, &s.g_old, &s.g_new, field_tesla);

    CopyGrid(&s.g_old, &s.g_new);
    printf("After GSA %e J\n", Hamiltonian(&s.g_old, field_tesla));
    PrintVecGridToFile("./output/new.vec", s.g_old.grid, s.g_old.param.rows, s.g_old.param.cols);

    ExportSimulatorFile(&s, "./output/export_sim.out");
    FreeSimulator(&s);
    return 0;
}

int main2()
{
    Grid g_old = InitGridFromFile("./input/teste.vec");
    GetGridParam("./input/input.in", &g_old.param);
    Grid g_new = InitNullGrid();
    CopyGrid(&g_new, &g_old);

    Vec field_joule = VecFrom(0.0, 0.0, 0.5 * g_old.param.dm * g_old.param.dm / g_old.param.exchange);
    Vec field_tesla = FieldJouleToTesla(field_joule, g_old.param.mu_s);

    PrintVecGridToFile("./output/old.vec", g_old.grid, g_old.param.rows, g_old.param.cols);
    printf("%e\n", field_tesla.z);
    
    printf("Before GSA %e J\n", Hamiltonian(&g_old, field_tesla));

    GSAParam gsap = {2.8, 2.2, 2.6, 2.0, 7, 1, 1};

    GSA(gsap, &g_old, &g_new, field_tesla);

    CopyGrid(&g_old, &g_new);
    printf("After GSA %e J\n", Hamiltonian(&g_old, field_tesla));
    PrintVecGridToFile("./output/new.vec", g_old.grid, g_old.param.rows, g_old.param.cols);

    FreeGrid(&g_old);
    FreeGrid(&g_new);
    
    return 0;
}