#include "atomistic_simulation.h"
#include <stdint.h>


//@TODO: PROPER ERROR CHECKING URGENT!!!
//@TODO: check time measuring when compiling it to win
int Stosic(void) {
    unsigned int rows = 64;
    unsigned int cols = 64;

    double lattice = 0.251e-9;
    double alpha = 0.037;
    double J = 29.0e-3 * QE;
    double dm = 1.5e-3 * QE;
    double ani = 0.293e-3 * QE;

    grid g = grid_init(rows, cols);
    grid_set_lattice(&g, lattice);
    grid_set_alpha(&g, alpha);
    grid_set_exchange(&g, J);
    grid_set_anisotropy(&g, (anisotropy){.dir=v3d_c(0.0, 0.0, 1.0), .ani = ani});
    double mu = 2.1 * MU_B;
    grid_set_mu(&g, mu);
    dm_interaction default_dm = (dm_interaction){.dmv_down = v3d_c(dm, 0.0, 0.0),
                                                 .dmv_up = v3d_c(-dm, 0.0, 0.0),
                                                 .dmv_left = v3d_c(0.0, -dm, 0.0),
                                                 .dmv_right = v3d_c(0.0, dm, 0.0)};

    grid_set_dm(&g, default_dm);
    for (unsigned int i = 0; i < rows * cols; ++i)
        g.m[i] = v3d_c(0, 0, -1);


    //v3d_create_skyrmion(g.m, g.gi.rows, g.gi.cols, 10, rows / 2, cols / 2, 1, 1, M_PI / 2.0);
    grid_create_skyrmion_at(&g, 10, 5, cols / 2, rows / 2, 1, 1, 0);
    g.gi.pbc = (pbc_rules){.pbc_x = 0, .pbc_y = 0, .m = v3d_c(0, 0, -1)};

    double dt = 0.01 * HBAR / (J * SIGN(J));
    //string current_func = str_is_cstr("current ret = (current){};\n"\
    //                                  "time -= 0.002 * NS;\n"\
    //                                  "ret.type = CUR_STT;\n"\
    //                                  "ret.stt.j = v3d_c(1e10, 10e10 * sin(time / NS * 5), 0.0);\n"\
    //                                  "ret.stt.beta = 0.0;\n"\
    //                                  "ret.stt.polarization = -1.0;\n"\
    //                                  "ret.stt.j = v3d_scalar(ret.stt.j, (time > 0));\n"\
    //                                  "return ret;");

    //string field_func = str_from_fmt("double Hz = %.15e;\n"\
    //                                 "return v3d_c(0.0, 0.0, Hz);", 0.5 * dm * dm / J * 1.0 / mu);



    logging_log(LOG_INFO, "Integration dt: %e", dt);
    integrate_params int_params = integrate_params_init();
    int_params.field_func = create_field_D2_over_J(v3d_c(0, 0, 0.5), J, dm, mu);
    int_params.duration = 20 * NS;
    int_params.interval_for_raw_grid = 1000;
    int_params.dt = dt;
    grid_renderer_integrate(&g, int_params, 1000, 1000);

    //str_free(&field_func);
    grid_free(&g);
    return 0;
}

struct data {
    double dm, J, ani;
    int size;
};

void set_pin(grid *g, uint64_t row, uint64_t col, void *dummy) {
    struct data *it = (struct data*)dummy;
    int center = 32;
    int dx = col - center;
    int dy = row - center;
    double d2 = dx * dx + dy * dy;
    grid_set_exchange_loc(g, row, col, it->J * 2);
    double angle = atan2(dy, dx) + M_PI;
    g->m[row * g->gi.cols + col] = v3d_c(0 * cos(angle), 0 * sin(angle), -1);
}

void set_pin2(grid *g, uint64_t row, uint64_t col, void *dummy) {
    UNUSED(dummy);
    g->m[row * g->gi.cols + col] = v3d_c(0, 0, -1);
}

int test(void) {
    unsigned int rows = 64;
    unsigned int cols = 64;

    double lattice = 0.5e-9;
    double J = 1.0e-3 * QE;
    double dm = 0.2 * J;
    double ani = 0.01 * J;
    double alpha = 0.3;
    //
    grid g = grid_init(rows, cols);
    grid_set_lattice(&g, lattice);
    grid_set_alpha(&g, alpha);
    grid_set_exchange(&g, J);
    grid_set_anisotropy(&g, (anisotropy){.dir=v3d_c(0.0, 0.0, 1.0), .ani = ani});

    double mu = g.gp->mu;

    dm_interaction default_dm = (dm_interaction){.dmv_down = v3d_c(0.0, dm, 0.0),
                                                 .dmv_up = v3d_c(0.0, -dm, 0.0),
                                                 .dmv_left = v3d_c(-dm, 0.0, 0.0),
                                                 .dmv_right = v3d_c(dm, 0.0, 0.0)};
    default_dm = dm_interfacial(dm);

    grid_set_dm(&g, default_dm);

    grid_uniform(&g, v3d_c(0, 0, 1));
    grid_create_skyrmionium_at(&g, 10, 10, cols / 2.0, rows / 2.0, 1, 1, 0);
    //grid_create_skyrmion_at(&g, 15, 3, 1 * cols / 5.0, rows / 2.0, -1, 1, 0);


    double dt = 0.01 * HBAR / (J * SIGN(J));
    double ratio = (double)rows / cols;
    logging_log(LOG_INFO, "Integration dt: %e", dt);
    integrate_params int_params = integrate_params_init();
    int_params.field_func = create_field_D2_over_J(v3d_c(0, 0, 0.5), J, dm, mu);
    int_params.duration = 200.2 * NS;
    int_params.interval_for_raw_grid = 1000;
    int_params.dt = dt;
    int_params.do_cluster = true;
    int_params.interval_for_cluster = 100;

    //int_params.current_func = str_is_cstr("current ret = (current){.type=CUR_SHE};\n"\
    //                                      "double x = gs.col - 0.5 * 64;\n"\
    //                                      "double y = gs.row - 0.5 * 64;\n"\
    //                                      "int ring = x * x + y * y <= 15 * 15 && x * x + y * y >= 13 * 13;\n"\
    //                                      "ret.she.p = v3d_scalar(v3d_normalize(v3d_c(x, y, 0)), 1000e10 * ring * (time < 40e-12));\n"\
    //                                      "ret.she.thickness = 0.5e-9;\n"\
    //                                      "ret.she.beta = 0;\n"\
    //                                      "ret.she.theta_sh = 1;\n"\
    //                                      "return ret;\n");

    grid_renderer_integrate(&g, int_params, 1000, 1000);

    double jx = 1e10;
    double jy = 1e10;
    int_params.current_func = create_current_she_ac(jx, v3d_c(jx, jy, 0), 100 / (200 * NS), 0);
    grid_renderer_integrate(&g, int_params, 1000, 1000);
    

    for (uint64_t i = 0; i < g.clusters.len; ++i) {
        cluster_center *it = &g.clusters.items[i];
        logging_log(LOG_INFO, "%e %e - %e %e %e", it->x, it->y, it->avg_m.x, it->avg_m.y, it->avg_m.z);
    }

    grid_free(&g);
    return 0;
}

struct data2 {
    double x0;
    double x1;
    double ani_0;
    double ani_1;
};

void do_stuff(grid *g, uint64_t row, uint64_t col, void *user) {
    struct data2 d = *(struct data2*)user;
    double d_ani = d.ani_1 - d.ani_0;
    double dx = d.x1 - d.x0;
    double ani = (col - d.x0) / dx * d_ani + d.ani_0;
    grid_set_anisotropy_loc(g, row, col, anisotropy_z_axis(ani));
}

int test2(void) {
    unsigned int rows = 128;
    unsigned int cols = 128;

    double lattice = 0.5e-9;
    double J = 1.0e-3 * QE;
    double dm = 0.3 * J * SIGN(J);
    double ani = 0.02 * J * SIGN(J);
    double alpha = 0.3;
    grid g = grid_init(rows, cols);
    grid_set_lattice(&g, lattice);
    grid_set_alpha(&g, alpha);
    grid_set_exchange(&g, J);
    grid_set_anisotropy(&g, (anisotropy){.dir=v3d_c(1.0, 0.0, 0), .ani = ani});

    double mu = g.gp->mu;
    dm_interaction default_dm = dm_bulk(dm);
    grid_set_dm(&g, default_dm);
    grid_uniform(&g, v3d_c(0, 0, 1));
    grid_create_skyrmion_at(&g, 5, 5, cols / 2, rows / 2, 1, 1, -M_PI);
    for (uint64_t i = 0; i < rows * cols; ++i)
        g.m[i] = v3d_c(g.m[i].z, g.m[i].x, g.m[i].y);


    double dt = 0.01 * HBAR / (J * SIGN(J));
    double ratio = (double)rows / cols;
    logging_log(LOG_INFO, "Integration dt: %e", dt);
    integrate_params int_params = integrate_params_init();
    //int_params.current_func = str_is_cstr("return (current){.type = CUR_SHE, .she.p = v3d_c(0, 0, -1000e6 / 1e-4 * (pow(gs.row - 128/ 2, 2.0) + pow(gs.col - 128/ 2, 2.0) <= pow(10, 2.0)) * (time < 0.2 * NS)), .she.thickness = 0.5e-9, .she.theta_sh=1, .she.beta = 0};");//create_current_she_dc(1000e6 / (1e-4), v3d_c(0, 0, 1), 0);
    int_params.field_func = create_field_D2_over_J(v3d_c(0.6, 0, 0.0), J, dm, mu);
    int_params.duration = 200.2 * NS;
    int_params.interval_for_raw_grid = 0;
    int_params.do_cluster = true;
    int_params.dt = dt;

    grid_renderer_integrate(&g, int_params, 1000, 1000);

    int_params.current_func = create_current_she_dc(-5e10, v3d_c(cos(26 / 180.0 * M_PI), sin(26 / 180.0 * M_PI), 0), 0);
    grid_renderer_integrate(&g, int_params, 1000, 1000);

    grid_free(&g);
    return 0;
}

int test3(void) {
    unsigned int rows = 64;
    unsigned int cols = 64;

    double lattice = 0.5e-9;
    double J = 1.0e-3 * QE;
    double dm = 0.00 * J * SIGN(J);
    double ani = 0.00 * J * SIGN(J);
    double alpha = 0.3;
    grid g = grid_init(rows, cols);
    grid_set_lattice(&g, lattice);
    grid_set_alpha(&g, alpha);
    grid_set_exchange(&g, J);
    grid_set_anisotropy(&g, (anisotropy){.dir=v3d_c(0.0, 0.0, 1), .ani = ani});

    double mu = g.gp->mu;
    dm_interaction default_dm = dm_bulk(dm);
    grid_set_dm(&g, default_dm);
    grid_uniform(&g, v3d_c(0, 0, 1));
    grid_create_biskyrmion_at(&g, 10, 20, cols / 2, rows / 2, 15, M_PI / 2.0, -1, 1, M_PI / 2);
    //grid_create_skyrmion_at(&g, 5, 5, cols / 2, rows / 2, 1, 1, -M_PI);

    double dt = 0.01 * HBAR / (J * SIGN(J));
    double ratio = (double)rows / cols;
    logging_log(LOG_INFO, "Integration dt: %e", dt);
    integrate_params int_params = integrate_params_init();
    int_params.field_func = create_field_D2_over_J(v3d_c(0.0, 0, 0.5), J, dm, mu);
    int_params.duration = 200.2 * NS;
    int_params.interval_for_raw_grid = 0;
    int_params.dt = dt;
    steps_per_frame = 100;

    grid_renderer_integrate(&g, int_params, 1000, 1000);

    //int_params.current_func = create_current_she_dc(-5e10, v3d_c(cos(26 / 180.0 * M_PI), sin(26 / 180.0 * M_PI), 0), 0);
    int_params.current_func = create_current_stt_dc(1e10, 0, 0);
    grid_renderer_integrate(&g, int_params, 1000, 1000);

    grid_free(&g);
    return 0;
}

int main(void) {
    test();
    //organize_clusters("./clusters.dat", "clusters_org.dat", 64 * 0.5e-9, 64 * 0.5e-9, 1e8, true);
    return 0;
}

