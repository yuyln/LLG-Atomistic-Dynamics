#include "atomistic_simulation.h"

double metric(grid *g, uint64_t i0, uint64_t j0, uint64_t k0, uint64_t i1, uint64_t j1, uint64_t k1, void *user_data) {
    UNUSED(user_data);
    v3d m0 = V_AT(g->m, i0, j0, k0, g->gi.rows, g->gi.cols);
    v3d m1 = V_AT(g->m, i1, j1, k1, g->gi.rows, g->gi.cols);
    m0.z = m0.z < 0.0? -1: 1;
    m1.z = m1.z < 0.0? -1: 1;
    return fabs(m1.z - m0.z) + (k0 != k1) * 1000;
}

void alter_magnetic(void) {
    int rows = 64;
    int cols = 64;
    grid g = grid_init(rows, cols, 2);
    double J1 = 1.0e-3 * QE;
    double J2 = 2e-3 * QE;
    double J3 = -J1 * 0.05;
    double D1 = 0.2 * J1;
    double D2 = D1;
    double K = 0.05 * J1;
    double mu = g.gp->mu;
    grid_set_alpha(&g, 0.3);
    exchange_interaction JA = (exchange_interaction){.J_left = J1, .J_right = J1,
                                                     .J_up = J2, .J_down = J2,
                                                     .J_front = J3, .J_back = J3};

    exchange_interaction JB = (exchange_interaction){.J_left = J2, .J_right = J2,
                                                     .J_up = J1, .J_down = J1,
                                                     .J_front = J3, .J_back = J3};
						     
    grid_set_anisotropy(&g, anisotropy_z_axis(K));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
	    dm_interaction D = dm_bulk(D1);
	    D.dmv_front = v3d_s(0);
	    D.dmv_back = v3d_s(0);
            V_AT(g.gp, i, j, 0, rows, cols).dm = D;
            V_AT(g.gp, i, j, 0, rows, cols).exchange = JA;
	    
	    D = dm_bulk(D2);
	    D.dmv_front = v3d_s(0);
	    D.dmv_back = v3d_s(0);
            V_AT(g.gp, i, j, 1, rows, cols).dm = D;
            V_AT(g.gp, i, j, 1, rows, cols).exchange = JB;
	}
    }
    
    grid_uniform(&g, v3d_c(0, 0, 1));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            V_AT(g.m, i, j, 0, rows, cols).z *= -1;
	}
    }
    V_AT(g.gp, rows / 2, cols / 2, 1, rows, cols).pin = (pinning){.pinned = 1, .dir = v3d_c(0, 0, -1)};
    V_AT(g.gp, rows / 2 + 1, cols / 2, 1, rows, cols).pin = (pinning){.pinned = 1, .dir = v3d_c(0, 0, -1)};
    V_AT(g.gp, rows / 2, cols / 2 + 1, 1, rows, cols).pin = (pinning){.pinned = 1, .dir = v3d_c(0, 0, -1)};
    V_AT(g.gp, rows / 2 - 1, cols / 2, 1, rows, cols).pin = (pinning){.pinned = 1, .dir = v3d_c(0, 0, -1)};
    V_AT(g.gp, rows / 2, cols / 2 - 1, 1, rows, cols).pin = (pinning){.pinned = 1, .dir = v3d_c(0, 0, -1)};

    V_AT(g.gp, rows / 2, cols / 2, 0, rows, cols).pin = (pinning){.pinned = 1, .dir = v3d_c(0, 0, 1)};
    V_AT(g.gp, rows / 2 + 1, cols / 2, 0, rows, cols).pin = (pinning){.pinned = 1, .dir = v3d_c(0, 0, 1)};
    V_AT(g.gp, rows / 2, cols / 2 + 1, 0, rows, cols).pin = (pinning){.pinned = 1, .dir = v3d_c(0, 0, 1)};
    V_AT(g.gp, rows / 2 - 1, cols / 2, 0, rows, cols).pin = (pinning){.pinned = 1, .dir = v3d_c(0, 0, 1)};
    V_AT(g.gp, rows / 2, cols / 2 - 1, 0, rows, cols).pin = (pinning){.pinned = 1, .dir = v3d_c(0, 0, 1)};
    
    g.gi.pbc.pbc_z = 0;
    integrate_params ip = integrate_params_init();
    ip.output_path = ".";
    ip.cluster_metric = metric;
    ip.dt = 0.01 * HBAR / J1;
    grid_renderer_integrate(&g, ip, 1000, 1000);
    for (int i = 0; i < rows * cols * 2; ++i)
        g.gp[i].pin = (pinning){0};

    ip.current_func = str_fmt_tmp("return (current){.type=CUR_STT, .stt.j = v3d_c(1e11, 0, 0), .stt.polarization = -1, .stt.beta = 1};");
    grid_renderer_integrate(&g, ip, 1000, 1000);

    grid_free(&g);
}

void non_reci(void) {
    int rows = 64;
    int cols = 64;
    int depth = 24;
    grid g = grid_init(rows, cols, depth);
    double J = 1.0e-3 * QE;
    double J_inter = -J * 0.5;
    double dm = 0.2 * J;
    double K = 0.005 * J;
    double mu = g.gp->mu;
    
    grid_set_alpha(&g, 0.3);
    grid_set_anisotropy(&g, anisotropy_z_axis(K));
    
    exchange_interaction Ji = isotropic_exchange(J);
    Ji.J_front = J_inter;
    Ji.J_back = J_inter;
    grid_set_exchange(&g, Ji);

    grid_uniform(&g, v3d_c(0, 0, 1));
    for (int k = 0; k < depth; ++k) {
	for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
		dm_interaction D = dm_interfacial(dm * (k % 2 == 0? -1: 1));
		V_AT(g.gp, i, j, k, rows, cols).dm = D;
		
		V_AT(g.gp, i, j, k, rows, cols).mu *= k % 2 == 0? 3: 1;
		
		V_AT(g.gp, i, j, k, rows, cols).ani.ani *= k % 2 == 0? 9: 1;
		
		V_AT(g.m, i, j, k, rows, cols).z *= k % 2 == 0? -1: 1;
	    }
	}
	V_AT(g.gp, rows / 2, cols / 2,     k, rows, cols).pin = (pinning){.pinned = 1, .dir = v3d_c(0, 0, k % 2 == 0? 1: -1)};
	V_AT(g.gp, rows / 2 + 1, cols / 2, k, rows, cols).pin = (pinning){.pinned = 1, .dir = v3d_c(0, 0, k % 2 == 0? 1: -1)};
	V_AT(g.gp, rows / 2, cols / 2 + 1, k, rows, cols).pin = (pinning){.pinned = 1, .dir = v3d_c(0, 0, k % 2 == 0? 1: -1)};
	V_AT(g.gp, rows / 2 - 1, cols / 2, k, rows, cols).pin = (pinning){.pinned = 1, .dir = v3d_c(0, 0, k % 2 == 0? 1: -1)};
	V_AT(g.gp, rows / 2, cols / 2 - 1, k, rows, cols).pin = (pinning){.pinned = 1, .dir = v3d_c(0, 0, k % 2 == 0? 1: -1)};
    }
    
    g.gi.pbc.pbc_z = 0;
    integrate_params ip = integrate_params_init();
    ip.cluster_metric = metric;
    ip.dt = 0.01 * HBAR / J;
    grid_renderer_integrate(&g, ip, 1000, 1000);
    for (int i = 0; i < rows * cols * depth; ++i)
        g.gp[i].pin = (pinning){0};

    ip.current_func = str_fmt_tmp("return (current){.type=CUR_STT, .stt.j = v3d_c(0e11, 0, 0), .stt.polarization = -1, .stt.beta = 1};");
    grid_renderer_integrate(&g, ip, 1000, 1000);

    grid_free(&g);
}

void conical_background(void) {
    int rows = 64;
    int cols = 64;
    int depth = 64;
    grid g = grid_init(rows, cols, depth);
    double J = 1.0e-3 * QE;
    double dm = 0.2 * J;
    double K = 0.0 * J;
    double mu = g.gp->mu;
    
    grid_set_alpha(&g, 0.3);
    grid_set_anisotropy(&g, anisotropy_z_axis(K));
    grid_set_dm(&g, dm_bulk(dm));
    
    exchange_interaction Ji = isotropic_exchange(J);
    grid_set_exchange(&g, Ji);
    g.gi.pbc.pbc_z = 0;

    grid_uniform(&g, v3d_c(0, 0, 1));
    for (int k = 0; k < depth; ++k) {
	V_AT(g.gp, rows / 2, cols / 2,     k, rows, cols).pin = (pinning){.pinned = 1, .dir = v3d_c(0, 0, -1)};
	V_AT(g.gp, rows / 2 + 1, cols / 2, k, rows, cols).pin = (pinning){.pinned = 1, .dir = v3d_c(0, 0, -1)};
	V_AT(g.gp, rows / 2, cols / 2 + 1, k, rows, cols).pin = (pinning){.pinned = 1, .dir = v3d_c(0, 0, -1)};
	V_AT(g.gp, rows / 2 - 1, cols / 2, k, rows, cols).pin = (pinning){.pinned = 1, .dir = v3d_c(0, 0, -1)};
	V_AT(g.gp, rows / 2, cols / 2 - 1, k, rows, cols).pin = (pinning){.pinned = 1, .dir = v3d_c(0, 0, -1)};
    }
    
    integrate_params ip = integrate_params_init();
    ip.dt = 0.01 * HBAR / J;
    ip.field_func = create_field_D2_over_J(v3d_c(0, 0, 0.8), J, dm, mu);
    ip.temperature_func = create_temperature(20);

    grid_renderer_integrate(&g, ip, 1000, 1000);

    
    for (int i = 0; i < rows * cols * depth; ++i)
        g.gp[i].pin = (pinning){0};

	ip.current_func = str_fmt_tmp("return (current){.type=CUR_STT, .stt.j = v3d_c(0e11, 0, 0), .stt.polarization = -1, .stt.beta = 1};");
	ip.temperature_func = NULL;//create_temperature(0);
    grid_renderer_integrate(&g, ip, 1000, 1000);

    grid_free(&g);
}

void apply_test(grid *g, uint64_t k, uint64_t y, uint64_t x, void *dummy) {
    logging_log(LOG_INFO, "%u %u %u", x, y, k);
    V_AT(g->gp, y, x, k, g->gi.rows, g->gi.cols).pin = (pinning){.pinned = 1, .dir = v3d_c(1, 0, 0)};
}
void test_in_line(void) {
    int rows = 32;
    int cols = 32;
    int depth = 32;
    grid g = grid_init(rows, cols, depth);
    double J = 1.0e-3 * QE;
    double dm = 0.2 * J;
    double K = 0.0 * J;
    double mu = g.gp->mu;
    
    grid_set_alpha(&g, 0.3);
    grid_set_anisotropy(&g, anisotropy_z_axis(K));
    grid_set_dm(&g, dm_bulk(dm));
    
    grid_set_exchange(&g, isotropic_exchange(J));
    grid_uniform(&g, v3d_c(0, 0, 1));

    grid_do_in_line(&g, v3d_c(0, 0, 0), v3d_c(cols / 2, rows / 2, 0), 2, apply_test, NULL);
    
    integrate_params ip = integrate_params_init();
    grid_renderer_integrate(&g, ip, 1000, 1000);
    
    grid_free(&g);
}

int main(void) {
    p_id = 1;
    steps_per_frame = 100;
    int rows = 32, cols = 32;
    grid g = grid_init(32, 32, 1);
    double J = g.gp->exchange.J_left;
    double D = 0.5 * J;
    double mu = g.gp->mu;
    double K = 0.05 * J;
    grid_set_anisotropy(&g, anisotropy_z_axis(K));
    grid_set_dm(&g, dm_interfacial(D));
    grid_uniform(&g, v3d_c(0, 0, 1));
    grid_create_skyrmion_at(&g, 5, 2, cols / 2, rows / 2, -1, 1, M_PI / 2);

    integrate_params ip = integrate_params_init();
    ip.field_func = create_field_D2_over_J(v3d_c(0, 0, 0.5), J, D, mu);
    ip.dt = 0.01 * HBAR / J;
    grid_renderer_integrate(&g, ip, 1000, 1000);
    ip.current_func = create_current_she_dc(5e10, v3d_c(0, -1, 0), 0);
    ip.current_func = create_current_stt_dc(5e10, 0, 0.6);
    grid_renderer_integrate(&g, ip, 1000, 1000);
    grid_free(&g);
    return 0;
    alter_magnetic();
    test_in_line();
    //conical_background();
    //non_reci();
}
