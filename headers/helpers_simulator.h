#ifndef __SIM
#define __SIM

#include "helpers.h"
#if defined(RK4)
static const char *integration_method = "RK4";
#elif defined(RK2)
static const char *integration_method = "RK2";
#elif defined(EULER)
static const char *integration_method = "EULER";
#else
static const char *integration_method = "???";
#endif

#define kernels_n sizeof(kernels) / sizeof(char *)
static const char *kernels[] = {"termal_step", "hamiltonian_gpu", "reset_gpu", "step_gpu", "gradient_step_gpu", "reset_v3d_gpu"};

void free_gpu(gpu_t *g) {
    for (uint64_t i = 0; i < g->n_kernels; ++i) {
        clw_print_cl_error(stderr, clReleaseKernel(g->kernels[i].kernel), "Error releasing kernel %s", g->kernels[i].name);
    }

    if (g->kernels)
        free(g->kernels);

    clw_print_cl_error(stderr, clReleaseProgram(g->program), "Error releasing program");
    clw_print_cl_error(stderr, clReleaseContext(g->ctx), "Error releasing context");
    clw_print_cl_error(stderr, clReleaseCommandQueue(g->queue), "Error releasing queue");

    for (uint64_t i = 0; i < g->n_devs; ++i) {
        clw_print_cl_error(stderr, clReleaseDevice(g->devs[i]), "Error releasing device[%zu]", i);
    }
    if (g->devs)
        free(g->devs);
    if (g->plats)
        free(g->plats);
}

simulator_t init_simulator(const char *path) {
    simulator_t ret = {0};
    grid_param_t param_tmp = {0};
    ret.doing_relax = false;
    find_grid_param_path(path, &param_tmp);

    region_param_t region_default = {0};
    region_default.exchange_mult = 1.0;
    region_default.dm_mult = 1.0;
    region_default.field_mult = 1.0;
    region_default.dm_type = param_tmp.dm_type;

    parser_context input_ctx = parser_init_context(global_parser_context.seps);
    parser_start(path, &input_ctx);

    ret.dt = parser_get_double("DT", 0.01, &input_ctx);
    ret.n_steps = (uint64_t)parser_get_ull("STEPS", 10, 3000000, &input_ctx);

    ret.write_to_file = (bool)parser_get_int("WRITE", 10, 0, &input_ctx);
    ret.write_cut = (uint64_t)parser_get_ull("CUT", 10, 8000, &input_ctx);
    ret.write_vel_charge_cut = (uint64_t)parser_get_ull("CUT_FOR_VEL_CHARGE", 10, 100, &input_ctx);
    ret.do_gsa = (bool)parser_get_int("GSA", 10, 0, &input_ctx);
    ret.do_relax = (bool)parser_get_int("RELAX", 10, 0, &input_ctx);
    ret.do_integrate = (bool)parser_get_int("INTEGRATE", 10, 1, &input_ctx);
    ret.calculate_energy = (bool)parser_get_int("CALCULATE_ENERGY", 10, 1, &input_ctx);
    ret.write_human = (bool)parser_get_int("WRITE_HUMAN", 10, 0, &input_ctx);
    ret.write_on_fly = (bool)parser_get_int("WRITE_ON_FLY", 10, 0, &input_ctx);
    ret.do_gradient = (bool)parser_get_int("GRADIENT", 10, 0, &input_ctx);
    ret.gradient_steps = parser_get_int("GRADIENT_STEPS", 10, ret.n_steps, &input_ctx);
    ret.dt_gradient = parser_get_double("GRADIENT_DT", ret.dt, &input_ctx);
    ret.alpha_gradient = parser_get_double("GRADIENT_ALPHA", 0, &input_ctx);
    ret.beta_gradient = parser_get_double("GRADIENT_BETA", 0, &input_ctx);
    ret.temp_gradient = parser_get_double("GRADIENT_TEMP", 10, &input_ctx);
    ret.factor_gradient = parser_get_double("GRADIENT_FACTOR",0.999, &input_ctx);
    ret.mass_gradient = parser_get_double("GRADIENT_MASS", 1.0, &input_ctx);

    ret.gsap.qA = parser_get_double("qA", 2.8, &input_ctx);
    ret.gsap.qV = parser_get_double("qV", 2.6, &input_ctx);
    ret.gsap.qT = parser_get_double("qT", 2.2, &input_ctx);
    ret.gsap.outer_loop = parser_get_ull("OUTER", 10, 10, &input_ctx);
    ret.gsap.inner_loop = parser_get_ull("INNER", 10, 100000, &input_ctx);
    ret.gsap.print_param = parser_get_ull("PRINTPARAM", 10, 10, &input_ctx);
    ret.gsap.T0 = parser_get_double("STARTTEMP", 2.00, &input_ctx);

    ret.n_cpu = (uint64_t)parser_get_ull("CPU", 10, 1, &input_ctx);


    char *local_file_ani_dir = NULL;
    char *local_file_pin_dir = NULL;
    char *local_file_regions_dir = NULL;
    char *local_file_grid_dir = NULL;


    if (parser_find_index_of_tag("FILE_ANISOTROPY", &input_ctx) > 0)
        local_file_ani_dir = strdup(input_ctx.state[parser_find_index_of_tag("FILE_ANISOTROPY", &input_ctx) + 1]);
    else
        fprintf(stderr, "Could not find FILE_ANISOTROPY, sample without anisotropy defects will be used\n");

    if (parser_find_index_of_tag("FILE_PINNING", &input_ctx) > 0)
        local_file_pin_dir = strdup(input_ctx.state[parser_find_index_of_tag("FILE_PINNING", &input_ctx) + 1]);
    else
        fprintf(stderr, "Could not find FILE_PINNING, sample without pinnings defects will be used\n");

    if (parser_find_index_of_tag("FILE_REGIONS", &input_ctx) > 0)
        local_file_regions_dir = strdup(input_ctx.state[parser_find_index_of_tag("FILE_REGIONS", &input_ctx) + 1]);
    else
        fprintf(stderr, "Could not find FILE_REGIONS, sample without different regions will be used\n");

    if (parser_find_index_of_tag("FILE_GRID", &input_ctx) > 0)
        local_file_grid_dir = strdup(input_ctx.state[parser_find_index_of_tag("FILE_GRID", &input_ctx) + 1]);
    else
        fprintf(stderr, "Could not find FILE_GRID, random starting sample will be used\n");

    anisotropy_t global_ani;
    global_ani.K_1 = parser_get_double("ANISOTROPY", 0.02, &input_ctx) * fabs(param_tmp.exchange);
    global_ani.dir.x = parser_get_double("ANI_X", 0, &input_ctx);
    global_ani.dir.y = parser_get_double("ANI_Y", 0, &input_ctx);
    global_ani.dir.z = parser_get_double("ANI_Z", 1.0, &input_ctx);
    global_ani.dir = v3d_normalize(global_ani.dir);

    if (!local_file_grid_dir)
        ret.g_old = init_grid_random(parser_get_int("ROWS", 10, 272, &input_ctx), parser_get_int("COLS", 10, 272, &input_ctx));
    else
        ret.g_old = init_grid_from_file(local_file_grid_dir);

    memcpy(&ret.g_old.param.exchange, &param_tmp.exchange, sizeof(grid_param_t) - (sizeof(int) * 2 + sizeof(uint64_t)));

    for (uint64_t i = 0; i < ret.g_old.param.total; ++i) {
        ret.g_old.ani[i] = global_ani;
        ret.g_old.regions[i] = region_default;
    }

    if (local_file_ani_dir) {
        parser_context anif_ctx = parser_init_context(global_parser_context.seps);
        parser_start(local_file_ani_dir, &anif_ctx);

        int index_data = parser_find_index_of_tag("Data", &anif_ctx);
        if (index_data < 0) {
            fprintf(stderr, "Tag \"Data\" not found on %s, empty anisotropy will be used\n", local_file_ani_dir);
            memset(ret.g_old.ani, 0, sizeof(anisotropy_t) * ret.g_old.param.total);
        } else {
            for (uint64_t I = index_data + 1; I < anif_ctx.n; I += 6) {
                int row = strtol(anif_ctx.state[I], NULL, 10);
                int col = strtol(anif_ctx.state[I + 1], NULL, 10);
                if (row < 0 || row >= ret.g_old.param.rows || col < 0 || col >= ret.g_old.param.cols)
                    continue;
                double dir_x = strtod(anif_ctx.state[I + 2], NULL);
                double dir_y = strtod(anif_ctx.state[I + 3], NULL);
                double dir_z = strtod(anif_ctx.state[I + 4], NULL);
                double K_1 = strtod(anif_ctx.state[I + 5], NULL) * param_tmp.exchange;
                ret.g_old.ani[row * ret.g_old.param.cols + col] = (anisotropy_t){K_1, v3d_c(dir_x, dir_y, dir_z)};
            }
        }
        parser_end(&anif_ctx);
    } else {
        memset(ret.g_old.ani, 0, sizeof(anisotropy_t) * ret.g_old.param.total);
    }

    if (local_file_pin_dir) {
        parser_context pinf_ctx = parser_init_context(global_parser_context.seps);
        parser_start(local_file_pin_dir, &pinf_ctx);

        int index_data = parser_find_index_of_tag("Data", &pinf_ctx);
        if (index_data < 0) {
            fprintf(stderr, "Tag \"Data\" not found on %s, empty pinning will be used\n", local_file_pin_dir);
            memset(ret.g_old.pinning, 0, sizeof(pinning_t) * ret.g_old.param.total);
        } else {
            for (uint64_t I = index_data + 1; I < pinf_ctx.n; I += 5) {
                int row = strtol(pinf_ctx.state[I], NULL, 10);
                int col = strtol(pinf_ctx.state[I + 1], NULL, 10);
                if (row < 0 || row >= ret.g_old.param.rows || col < 0 || col >= ret.g_old.param.cols)
                    continue;
                double dir_x = strtod(pinf_ctx.state[I + 2], NULL);
                double dir_y = strtod(pinf_ctx.state[I + 3], NULL);
                double dir_z = strtod(pinf_ctx.state[I + 4], NULL);
                ret.g_old.pinning[row * ret.g_old.param.cols + col] = (pinning_t){1, v3d_c(dir_x, dir_y, dir_z)};
            }
        }
        parser_end(&pinf_ctx);
    } else {
        memset(ret.g_old.pinning, 0, sizeof(pinning_t) * ret.g_old.param.total);
    }

    if (local_file_regions_dir) {
        parser_context regionf_ctx = parser_init_context(global_parser_context.seps);
        parser_start(local_file_regions_dir, &regionf_ctx);
        int index_data = parser_find_index_of_tag("Data", &regionf_ctx);
        if (index_data < 0) {
            fprintf(stderr, "Tag \"Data\" not found on %s\n", local_file_regions_dir);
            memset(ret.g_old.regions, 0, sizeof(region_param_t) * ret.g_old.param.total);
        } else {
            for (uint64_t I = index_data + 1; I < regionf_ctx.n; I += 6) {
                int row = strtol(regionf_ctx.state[I], NULL, 10);
                int col = strtol(regionf_ctx.state[I + 1], NULL, 10);
                if (row < 0 || row >= ret.g_old.param.rows || col < 0 || col >= ret.g_old.param.cols)
                    continue;
                ret.g_old.regions[row * ret.g_old.param.cols + col].exchange_mult = strtod(regionf_ctx.state[I + 2], NULL);
                ret.g_old.regions[row * ret.g_old.param.cols + col].dm_mult = strtod(regionf_ctx.state[I + 3], NULL);
                ret.g_old.regions[row * ret.g_old.param.cols + col].dm_type = (int)strtol(regionf_ctx.state[I + 4], NULL, 10);
                ret.g_old.regions[row * ret.g_old.param.cols + col].field_mult = strtod(regionf_ctx.state[I + 5], NULL);
            }
        }
        parser_end(&regionf_ctx);
    } else {
        memset(ret.g_old.regions, 0, sizeof(region_param_t) * ret.g_old.param.total);
    }

    ret.grid_out_file = (v3d *)calloc(ret.write_to_file * ret.n_steps * ret.g_old.param.total / ret.write_cut, sizeof(v3d));
    ret.velxy_Ez = (v3d *)calloc(ret.n_steps / ret.write_vel_charge_cut, sizeof(v3d));
    ret.pos_xy = (v3d *)calloc(ret.n_steps / ret.write_vel_charge_cut, sizeof(v3d));
    ret.avg_mag = (v3d *)calloc(ret.n_steps / ret.write_vel_charge_cut, sizeof(v3d));
    ret.chpr_chim = (v3d *)calloc(ret.n_steps / ret.write_vel_charge_cut, sizeof(v3d));
    printf("Size of grid out file in MB: %f\n", (ret.write_to_file || ret.write_on_fly) * ret.n_steps * ret.g_old.param.total / ret.write_cut * sizeof(v3d) / 1.0e6);

    ret.g_old.param.total_time = ret.dt * ret.n_steps;
    grid_copy(&ret.g_new, &ret.g_old);
    ret.use_gpu = (bool)parser_get_int("GPU", 10, 1, &input_ctx);
    if (ret.use_gpu) {
        ret.gpu.i_plat = parser_get_int("PLAT", 10, 0, &input_ctx);
        ret.gpu.i_dev = parser_get_int("DEV", 10, 0, &input_ctx);
        ret.gpu.plats = clw_init_platforms(&ret.gpu.n_plats);

        for (uint64_t i = 0; i < ret.gpu.n_plats; ++i)
            clw_get_platform_info(stdout, ret.gpu.plats[i], i);

        ret.gpu.devs = clw_init_devices(ret.gpu.plats[ret.gpu.i_plat], &ret.gpu.n_devs);
        for (uint64_t i = 0; i < ret.gpu.n_devs; ++i)
            clw_get_device_info(stdout, ret.gpu.devs[i], i);

        ret.gpu.ctx = clw_init_context(ret.gpu.devs, ret.gpu.n_devs);
        ret.gpu.queue = clw_init_queue(ret.gpu.ctx, ret.gpu.devs[ret.gpu.i_dev]);
        ret.gpu.program = clw_init_program_source(ret.gpu.ctx, kernel_data);

        char *comp_opt;
        const char* compile_line = "-DROWS=%d -DCOLS=%d -DTOTAL=%zu -DOPENCLCOMP -D%s -cl-nv-verbose";


        uint64_t comp_opt_size = snprintf(NULL, 0, compile_line, ret.g_old.param.rows, ret.g_old.param.cols, ret.g_old.param.total, integration_method) + 1;
        comp_opt = (char *)calloc(comp_opt_size, 1);
        snprintf(comp_opt, comp_opt_size, compile_line, ret.g_old.param.rows, ret.g_old.param.cols, ret.g_old.param.total, integration_method);
        comp_opt[comp_opt_size - 1] = '\0';

        printf("Compile OpenCL: %s\n", comp_opt);
        cl_int err = clw_build_program(ret.gpu.program, ret.gpu.n_devs, ret.gpu.devs, comp_opt);
        clw_get_program_build_info(stdout, ret.gpu.program, ret.gpu.devs[ret.gpu.i_dev], err);

        free(comp_opt);
        ret.g_old_buffer = clw_create_buffer(find_grid_size_bytes(&ret.g_old), ret.gpu.ctx, CL_MEM_READ_WRITE);
        ret.g_new_buffer = clw_create_buffer(find_grid_size_bytes(&ret.g_new), ret.gpu.ctx, CL_MEM_READ_WRITE);
        full_grid_write_buffer(ret.gpu.queue, ret.g_old_buffer, &ret.g_old);
        full_grid_write_buffer(ret.gpu.queue, ret.g_new_buffer, &ret.g_new);
        ret.gpu.kernels = clw_init_kernels(ret.gpu.program, kernels, kernels_n);
    }


    parser_end(&input_ctx);

    {
        printf("------------------------SIMULATION INFO-----------------------\n");
        printf("N CPUS                      = %zu\n", ret.n_cpu);
        printf("USE GPU                     = %d\n\n", ret.use_gpu);

        printf("DO GSA                      = %d\n", ret.do_gsa);
        printf("DO RELAX                    = %d\n", ret.do_relax);
        printf("DO GRADIENT                 = %d\n", ret.do_gradient);
        printf("DO INTEGRATE                = %d\n\n", ret.do_integrate);

        printf("DT NORMALIZED               = %.15e\n", ret.dt);
        printf("DT REAL                     = %.15e ns\n", ret.dt * HBAR / fabs(ret.g_old.param.exchange) / 1.0e-9);
        printf("INTEGRATIOn REAL TIME       = %.15e ns\n", ret.dt * HBAR / fabs(ret.g_old.param.exchange) * ret.n_steps / 1.0e-9);
        printf("INTEGRATION STEPS           = %zu\n\n", ret.n_steps);

        printf("WRITE TO FILE               = %d\n", ret.write_to_file);
        printf("CUT FOR WRITING             = %zu\n", ret.write_cut);
        printf("CUT FOR WRITING VEL         = %zu\n", ret.write_vel_charge_cut);
        printf("CALCULATE ENERGY EVOLUTION  = %d\n", ret.calculate_energy);
        printf("WRITE IN HUMAN FORM         = %d\n", ret.write_human);
        printf("WRITE DURING INTEGRATION    = %d\n\n", ret.write_on_fly);

        printf("GRADIENT STEPS              = %zu\n", ret.gradient_steps);
        printf("DT GRADIENT (UNITLESS)      = %.15e\n", ret.dt_gradient);
        printf("ALPHA GRADIENT              = %.15e\n", ret.alpha_gradient);
        printf("BETA GRADIENT               = %.15e\n", ret.beta_gradient);
        printf("TEMPERATURE GRADIENT        = %.15e\n", ret.temp_gradient);
        printf("TEMPERATURE FACTOR GRADIENT = %.15e\n", ret.factor_gradient);
        printf("MASS GRADIENT               = %.15e\n", ret.mass_gradient);

        printf("GSA qA                      = %.15e\n", ret.gsap.qA);
        printf("GSA qV                      = %.15e\n", ret.gsap.qV);
        printf("GSA qT                      = %.15e\n", ret.gsap.qT);
        printf("GSA OUT LOOP                = %zu\n", ret.gsap.outer_loop);
        printf("GSA INNER LOOP              = %zu\n", ret.gsap.inner_loop);
        printf("GSA PRINT PARAM             = %zu\n", ret.gsap.print_param);
        printf("GSA T0                      = %.15e\n\n", ret.gsap.T0);

        printf("RANDOM START GRID           = %d\n", local_file_grid_dir == NULL);
        if (local_file_grid_dir)
            printf("GRID STARTING FILE          = %s\n", local_file_grid_dir);
        printf("GRID ROWS x COLS            = %d x %d\n\n", ret.g_old.param.rows, ret.g_old.param.cols);



        printf("EXCHANGE                    = %.15e Joule = %.15e eV\n", ret.g_old.param.exchange, ret.g_old.param.exchange / QE);
        printf("DM                          = %.15e Joule = %.15e eV = %.15e * J\n", ret.g_old.param.dm, ret.g_old.param.dm / QE, ret.g_old.param.dm / fabs(ret.g_old.param.exchange));
        printf("LATTICE PARAMETER           = %.15e nm\n", ret.g_old.param.lattice / 1.0e-9);
        printf("CUBIC ANISOTROPY            = %.15e Joule = %.15e eV = %.15e * J\n", ret.g_old.param.cubic_ani, ret.g_old.param.cubic_ani / QE, ret.g_old.param.cubic_ani / fabs(ret.g_old.param.exchange));
        printf("AXIAL ANISOTROPY            = %.15e Joule = %.15e eV = %.15e * J\n", global_ani.K_1, global_ani.K_1 / QE, global_ani.K_1 / fabs(ret.g_old.param.exchange));
        printf("AXIAL ANISOTROPY            = (%.15e, %.15e, %.15e)\n", global_ani.dir.x, global_ani.dir.y, global_ani.dir.z);
        printf("LANDE                       = %.15e\n", ret.g_old.param.lande);
        printf("AVERAGE SPIN                = %.15e\n", ret.g_old.param.avg_spin);
        printf("SPIN MOMENTUM               = %.15e\n", ret.g_old.param.mu_s);
        printf("GILBERT DAMPING             = %.15e\n", ret.g_old.param.alpha);
        printf("GAMMA                       = %.15e\n", ret.g_old.param.gamma);

        printf("DM TYPE                     = %d\n", ret.g_old.param.dm_type);
        printf("PBC TYPE                    = %d\n", ret.g_old.param.pbc.pbc_type);
        printf("PBC DIR                     = (%.15e, %.15e, %.15e)\n", ret.g_old.param.pbc.dir.x, ret.g_old.param.pbc.dir.y, ret.g_old.param.pbc.dir.z);
        printf("--------------------------------------------------------------\n");
    }

    free(local_file_grid_dir);
    free(local_file_ani_dir);
    free(local_file_pin_dir);
    free(local_file_regions_dir);

#if !(defined(RK4) || defined(RK2) || defined(EULER))
    fprintf(stderr, "Invalid integration\n");
    exit(1);
#endif

    return ret;
}

void free_simulator(simulator_t *s) {
    if (s->grid_out_file) {
        free(s->grid_out_file);
        s->grid_out_file = NULL;
    }

    if (s->velxy_Ez) {
        free(s->velxy_Ez);
        s->velxy_Ez = NULL;
    }

    if (s->pos_xy) {
        free(s->pos_xy);
        s->pos_xy = NULL;
    }

    if (s->avg_mag) {
        free(s->avg_mag);
        s->avg_mag = NULL;
    }

    if (s->chpr_chim) {
        free(s->chpr_chim);
        s->chpr_chim = NULL;
    }

    grid_free(&s->g_old);
    grid_free(&s->g_new);
    if (s->use_gpu) {
        clw_print_cl_error(stderr, clReleaseMemObject(s->g_old_buffer), "Error releasing g_old buffer");
        clw_print_cl_error(stderr, clReleaseMemObject(s->g_new_buffer), "Error releasing g_new buffer");
        free_gpu(&s->gpu);
    }
}

void export_simulator(simulator_t *s, FILE *file) {
    // double J_abs = s->g_old.param.exchange * (s->g_old.param.exchange < 0? -1.0: 1.0);
    double J_abs = fabs(s->g_old.param.exchange);
    fprintf(file, "Times Steps: %zu\n", s->n_steps);
    fprintf(file, "Time Step: %e\n", s->dt);
    fprintf(file, "Total Time Real: %e\n\n\n", s->dt * s->n_steps * HBAR / J_abs);

    fprintf(file, "Exchange: %e J\n", s->g_old.param.exchange);
    fprintf(file, "DMI: %e J\n", s->g_old.param.dm);
    fprintf(file, "Cubic anisotropy_t: %e J\n\n", s->g_old.param.cubic_ani);
    fprintf(file, "MU_S: %e J/T\n", s->g_old.param.mu_s);
    fprintf(file, "Average Spin: %e \n", s->g_old.param.avg_spin);
    fprintf(file, "Lande: %e \n\n", s->g_old.param.lande);

    fprintf(file, "Alpha: %e \n", s->g_old.param.alpha);
    fprintf(file, "Gamma: %e \n\n", s->g_old.param.gamma);

    fprintf(file, "Integration: %s\n", integration_method);
    fprintf(file, "Write to File: %d\n", s->write_to_file);
    fprintf(file, "Write to File Cut: %zu\n", s->write_cut);
}

void export_simulator_path(simulator_t *s, const char *path) {
    FILE *file = file_open(path, "wb", 1);
    export_simulator(s, file);
    fclose(file);
}

void dump_write_grid(const char *file_path, simulator_t *s) {
    FILE *f = file_open(file_path, "wb", 1);

    fwrite(&s->g_old.param.rows, sizeof(s->g_old.param.rows), 1, f);
    fwrite(&s->g_old.param.cols, sizeof(s->g_old.param.cols), 1, f);
    int n_steps = s->n_steps / s->write_cut;
    fwrite(&n_steps, 1, sizeof(int), f);

    uint64_t ss = s->write_to_file * s->n_steps * s->g_old.param.total / s->write_cut * sizeof(v3d);
    fwrite(s->grid_out_file, 1, ss, f);

    fclose(f);
}

void DumpWriteChargegrid_t(const char *file_path, simulator_t *s) {
    FILE *f = file_open(file_path, "wb", 1);

    int rows = s->g_old.param.rows;
    int cols = s->g_old.param.cols;
    fwrite(&rows, sizeof(rows), 1, f);
    fwrite(&cols, sizeof(cols), 1, f);
    int n_steps = s->n_steps / s->write_cut;
    fwrite(&n_steps, 1, sizeof(int), f);

    uint64_t ss = s->write_to_file * s->n_steps * s->g_old.param.total / s->write_cut * sizeof(double);

    double *charge_total = calloc(s->write_to_file * s->n_steps * s->g_old.param.total / s->write_cut, sizeof(double));
    for (int i = 0; i < n_steps; ++i) {
        for (uint64_t I = 0; I < (uint64_t)(rows * cols); ++I) {
            charge_total[i * rows * cols + I] = charge(I, &s->grid_out_file[i * rows * cols], rows, cols, s->g_old.param.pbc);
        }
    }
    fwrite(charge_total, ss, 1, f);

    fclose(f);
    free(charge_total);
}
#endif
