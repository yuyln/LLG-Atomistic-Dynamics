#include <helpers.h>
#include <gsa.h>

int main()
{
    Grid g_old = InitGridFromFile("./input/teste.vec");
    GetGridParam("./input/gridparam.in", &g_old.param);
    Grid g_new = InitNullGrid();
    CopyGrid(&g_new, &g_old);

    Vec field_joule = VecFrom(0.0, 0.0, 0.5 * g_old.param.dm * g_old.param.dm / g_old.param.exchange);
    Vec field_tesla = FieldJouleToTesla(field_joule, g_old.param.mu_s);

    PrintVecGridToFile("./output/old.vec", g_old.grid, g_old.param.rows, g_old.param.cols);
    printf("%e\n", field_tesla.z);
    
    printf("Before GSA %e J\n", Hamiltonian(&g_old, field_tesla));

    GSAParam gsap = {2.8, 2.2, 2.6, 2.0, 700000, 10, 1};

    GSA(gsap, &g_old, &g_new, field_tesla);

    CopyGrid(&g_old, &g_new);
    printf("After GSA %e J\n", Hamiltonian(&g_old, field_tesla));
    PrintVecGridToFile("./output/new.vec", g_old.grid, g_old.param.rows, g_old.param.cols);

    FreeGrid(&g_old);
    FreeGrid(&g_new);
    
    return 0;
}