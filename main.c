#include <helpers.h>
#include <gsa.h>
#include <simulator.h>

int main()
{
    Simulator s = InitSimulator("./input/input.in");
    Vec field_joule = VecFrom(0.0, 0.0, 0.5 * s.g_old.param.dm * s.g_old.param.dm / s.g_old.param.exchange);
    Vec field_tesla = FieldJouleToTesla(field_joule, s.g_old.param.mu_s);

    cl_mem ham_buffer = CreateBuffer(s.g_old.param.total * sizeof(double), s.gpu.ctx, CL_MEM_READ_WRITE);
    double* ham_gpu = (double*)calloc(s.g_old.param.total, sizeof(double));
    SetKernelArg(s.gpu.kernels[1], 0, sizeof(cl_mem), &s.g_old_buffer);
    SetKernelArg(s.gpu.kernels[1], 1, sizeof(cl_mem), &ham_buffer);
    SetKernelArg(s.gpu.kernels[1], 2, sizeof(Vec), &field_tesla);
    
    size_t global = s.g_old.param.total;
    size_t local = gcd(global, 512);
    EnqueueND(s.gpu.queue, s.gpu.kernels[1], 1, NULL, &global, &local);
    Finish(s.gpu.queue);
    ReadBuffer(ham_buffer, ham_gpu, sizeof(double) * s.g_old.param.total, 0, s.gpu.queue);

    double ham = 0.0;
    for (size_t I = 0; I < s.g_old.param.total; ++I)
        ham += ham_gpu[I];
    
    printf("GPU: %e\nCPU: %e\n", ham, Hamiltonian(&s.g_old, field_tesla));


    free(ham_gpu);
    PrintCLError(stderr, clReleaseMemObject(ham_buffer), "Error releasing ham_buffer");
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