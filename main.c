#include <helpers.h>
#include <gsa.h>

int main()
{
    Grid g = InitGridFromFile("./input/teste.vec");
    Grid g2 = InitGridRandom(10, 10);

    PrintVecGridToFile("./output/g.vec", g.grid, g.param.rows, g.param.cols);
    PrintVecGridToFile("./output/g2.vec", g2.grid, g2.param.rows, g2.param.cols);
    printf("%e\n", Hamiltonian(&g2, VecFromScalar(0.0)));

    GSAParam gsap = {0};
    GSA(gsap, &g, &g2);

    CopyGrid(&g2, &g);
    FreeGrid(&g);
    FreeGrid(&g2);
    
    return 0;
}