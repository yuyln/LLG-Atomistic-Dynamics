#ifndef __SIM
#define __SIM

#include <helpers.h>
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
static const char *kernels[] = {"TermalStep", "HamiltonianGPU", "Reset", "StepGPU", "GradientStep", "ResetVec"};

void FreeGPU(GPU *g)
{
    for (size_t i = 0; i < g->n_kernels; ++i)
    {
        PrintCLError(stderr, clReleaseKernel(g->kernels[i].kernel), "Error releasing kernel %s", g->kernels[i].name);
    }

    if (g->kernels)
        free(g->kernels);

    PrintCLError(stderr, clReleaseProgram(g->program), "Error releasing program");
    PrintCLError(stderr, clReleaseContext(g->ctx), "Error releasing context");
    PrintCLError(stderr, clReleaseCommandQueue(g->queue), "Error releasing queue");

    for (size_t i = 0; i < g->n_devs; ++i)
    {
        PrintCLError(stderr, clReleaseDevice(g->devs[i]), "Error releasing device[%zu]", i);
    }
    if (g->devs)
        free(g->devs);
    if (g->plats)
        free(g->plats);
}

Simulator InitSimulator(const char *path)
{
    Simulator ret = {0};
    GridParam param_tmp = {0};
    ret.doing_relax = false;
    GetGridParam(path, &param_tmp);

    RegionParam region_default = {0};
    region_default.exchange_mult = 1.0;
    region_default.dm_mult = 1.0;
    region_default.field_mult = 1.0;
    region_default.dm_type = param_tmp.dm_type;

    StartParse(path);

    ret.dt = GetValueDouble("DT");
    ret.n_steps = (size_t)GetValueULLInt("STEPS", 10);

    ret.write_to_file = (bool)GetValueInt("WRITE", 10);
    ret.write_cut = (size_t)GetValueULLInt("CUT", 10);
    ret.write_vel_charge_cut = (size_t)GetValueULLInt("CUT_FOR_VEL_CHARGE", 10);
    ret.do_gsa = (bool)GetValueInt("GSA", 10);
    ret.do_relax = (bool)GetValueInt("RELAX", 10);
    ret.do_integrate = (bool)GetValueInt("INTEGRATE", 10);
    ret.calculate_energy = (bool)GetValueInt("CALCULATE_ENERGY", 10);
    ret.write_human = (bool)GetValueInt("WRITE_HUMAN", 10);
    ret.write_on_fly = (bool)GetValueInt("WRITE_ON_FLY", 10);
    ret.do_gradient = (bool)GetValueInt("GRADIENT", 10);
    ret.gradient_steps = GetValueInt("GRADIENT_STEPS", 10);
    ret.dt_gradient = GetValueDouble("GRADIENT_DT");
    ret.alpha_gradient = GetValueDouble("GRADIENT_ALPHA");
    ret.beta_gradient = GetValueDouble("GRADIENT_BETA");
    ret.temp_gradient = GetValueDouble("GRADIENT_TEMP");
    ret.factor_gradient = GetValueDouble("GRADIENT_FACTOR");
    ret.mass_gradient = GetValueDouble("GRADIENT_MASS");

    if (ret.gradient_steps == 0)
	ret.gradient_steps = ret.n_steps;
    if (ret.dt_gradient == 0)
	ret.dt_gradient = ret.dt;

    if (ret.do_gsa)
    {
        ret.gsap.qA = GetValueDouble("qA");
        ret.gsap.qV = GetValueDouble("qV");
        ret.gsap.qT = GetValueDouble("qT");
        ret.gsap.outer_loop = GetValueULLInt("OUTER", 10);
        ret.gsap.inner_loop = GetValueULLInt("INNER", 10);
        ret.gsap.print_param = GetValueULLInt("PRINTPARAM", 10);
        ret.gsap.T0 = GetValueDouble("STARTTEMP");
    }

    ret.n_cpu = (size_t)GetValueULLInt("CPU", 10);

    bool start_random = (bool)GetValueInt("RANDOM_START_GRID", 10);
    char *local_file_dir = strdup(parser_global_state[FindIndexOfTag("FILE_GRID") + 1]);

    if (FindIndexOfTag("FILE_ANISOTROPY") < 0)
    {
        fprintf(stderr, "Must provide path do anisotropy file, even if its empty");
        exit(1);
    }

    if (FindIndexOfTag("FILE_PINNING") < 0)
    {
        fprintf(stderr, "Must provide path do pinning file, even if its empty");
        exit(1);
    }

    if (FindIndexOfTag("FILE_REGIONS") < 0)
    {
        fprintf(stderr, "Must provide path do regions file, even if its empty");
        exit(1);
    }

    char *local_file_ani_dir = strdup(parser_global_state[FindIndexOfTag("FILE_ANISOTROPY") + 1]);
    char *local_file_pin_dir = strdup(parser_global_state[FindIndexOfTag("FILE_PINNING") + 1]);
    char *local_file_regions_dir = strdup(parser_global_state[FindIndexOfTag("FILE_REGIONS") + 1]);

    Anisotropy global_ani;
    global_ani.K_1 = GetValueDouble("ANISOTROPY") * param_tmp.exchange;
    global_ani.dir.x = GetValueDouble("ANI_X");
    global_ani.dir.y = GetValueDouble("ANI_Y");
    global_ani.dir.z = GetValueDouble("ANI_Z");
    global_ani.dir = VecNormalize(global_ani.dir);

    if (start_random)
        ret.g_old = InitGridRandom(GetValueInt("ROWS", 10), GetValueInt("COLS", 10));
    else
        ret.g_old = InitGridFromFile(local_file_dir);

    memcpy(&ret.g_old.param.exchange, &param_tmp.exchange, sizeof(GridParam) - (sizeof(int) * 2 + sizeof(size_t)));

    for (size_t i = 0; i < ret.g_old.param.total; ++i)
    {
        ret.g_old.ani[i] = global_ani;
        ret.g_old.regions[i] = region_default;
    }

    StartParse(local_file_ani_dir);

    int index_data = FindIndexOfTag("Data");
    if (index_data < 0)
    {
        fprintf(stderr, "Tag \"Data\" not found on %s\n", local_file_ani_dir);
        exit(1);
    }
    for (size_t I = index_data + 1; I < parser_global_n; I += 6)
    {
        int row = strtol(parser_global_state[I], NULL, 10);
        int col = strtol(parser_global_state[I + 1], NULL, 10);
	if (row < 0 || row >= ret.g_old.param.rows || col < 0 || col >= ret.g_old.param.cols)
	    continue;
        double dir_x = strtod(parser_global_state[I + 2], NULL);
        double dir_y = strtod(parser_global_state[I + 3], NULL);
        double dir_z = strtod(parser_global_state[I + 4], NULL);
        double K_1 = strtod(parser_global_state[I + 5], NULL) * param_tmp.exchange;
        ret.g_old.ani[row * ret.g_old.param.cols + col] = (Anisotropy){K_1, VecFrom(dir_x, dir_y, dir_z)};
    }

    EndParse();

    StartParse(local_file_pin_dir);

    index_data = FindIndexOfTag("Data");
    if (index_data < 0)
    {
        fprintf(stderr, "Tag \"Data\" not found on %s\n", local_file_pin_dir);
        exit(1);
    }
    for (size_t I = index_data + 1; I < parser_global_n; I += 5)
    {
        int row = strtol(parser_global_state[I], NULL, 10);
        int col = strtol(parser_global_state[I + 1], NULL, 10);
	if (row < 0 || row >= ret.g_old.param.rows || col < 0 || col >= ret.g_old.param.cols)
	    continue;
        double dir_x = strtod(parser_global_state[I + 2], NULL);
        double dir_y = strtod(parser_global_state[I + 3], NULL);
        double dir_z = strtod(parser_global_state[I + 4], NULL);
        ret.g_old.pinning[row * ret.g_old.param.cols + col] = (Pinning){1, VecFrom(dir_x, dir_y, dir_z)};
    }

    EndParse();

    StartParse(local_file_regions_dir);
    index_data = FindIndexOfTag("Data");
    if (index_data < 0)
    {
        fprintf(stderr, "Tag \"Data\" not found on %s\n", local_file_regions_dir);
        exit(1);
    }

    for (size_t I = index_data + 1; I < parser_global_n; I += 6)
    {
        int row = strtol(parser_global_state[I], NULL, 10);
        int col = strtol(parser_global_state[I + 1], NULL, 10);
	if (row < 0 || row >= ret.g_old.param.rows || col < 0 || col >= ret.g_old.param.cols)
	    continue;
        ret.g_old.regions[row * ret.g_old.param.cols + col].exchange_mult = strtod(parser_global_state[I + 2], NULL);
        ret.g_old.regions[row * ret.g_old.param.cols + col].dm_mult = strtod(parser_global_state[I + 3], NULL);
        ret.g_old.regions[row * ret.g_old.param.cols + col].dm_type = (int)strtol(parser_global_state[I + 4], NULL, 10);
        ret.g_old.regions[row * ret.g_old.param.cols + col].field_mult = strtod(parser_global_state[I + 5], NULL);
    }
    EndParse();

    ret.grid_out_file = (Vec *)calloc(ret.write_to_file * ret.n_steps * ret.g_old.param.total / ret.write_cut, sizeof(Vec));
    ret.velxy_Ez = (Vec *)calloc(ret.n_steps / ret.write_vel_charge_cut, sizeof(Vec));
    ret.pos_xy = (Vec *)calloc(ret.n_steps / ret.write_vel_charge_cut, sizeof(Vec));
    ret.avg_mag = (Vec *)calloc(ret.n_steps / ret.write_vel_charge_cut, sizeof(Vec));
    ret.chpr_chim = (Vec *)calloc(ret.n_steps / ret.write_vel_charge_cut, sizeof(Vec));
    printf("Size of grid out file in MB: %f\n", (ret.write_to_file || ret.write_on_fly) * ret.n_steps * ret.g_old.param.total / ret.write_cut * sizeof(Vec) / 1.0e6);
    free(local_file_dir);
    free(local_file_ani_dir);
    free(local_file_pin_dir);
    free(local_file_regions_dir);
    StartParse(path);

    ret.g_old.param.total_time = ret.dt * ret.n_steps;
    CopyGrid(&ret.g_new, &ret.g_old);
    ret.use_gpu = (bool)GetValueInt("GPU", 10);
    if (ret.use_gpu)
    {
        ret.gpu.i_plat = GetValueInt("PLAT", 10);
        ret.gpu.i_dev = GetValueInt("DEV", 10);
        ret.gpu.plats = InitPlatforms(&ret.gpu.n_plats);

        for (size_t i = 0; i < ret.gpu.n_plats; ++i)
            PlatformInfo(stdout, ret.gpu.plats[i], i);

        ret.gpu.devs = InitDevices(ret.gpu.plats[ret.gpu.i_plat], &ret.gpu.n_devs);
        for (size_t i = 0; i < ret.gpu.n_devs; ++i)
            DeviceInfo(stdout, ret.gpu.devs[i], i);

        ret.gpu.ctx = InitContext(ret.gpu.devs, ret.gpu.n_devs);
        ret.gpu.queue = InitQueue(ret.gpu.ctx, ret.gpu.devs[ret.gpu.i_dev]);
        ret.gpu.program = InitProgramSource(ret.gpu.ctx, kernel_data);

        char *comp_opt;


        size_t comp_opt_size = snprintf(NULL, 0, "-I ./headers -DROWS=%d -DCOLS=%d -DTOTAL=%zu -DOPENCLCOMP -D%s -cl-nv-verbose", ret.g_old.param.rows, ret.g_old.param.cols, ret.g_old.param.total, integration_method) + 1;
        comp_opt = (char *)calloc(comp_opt_size, 1);
        snprintf(comp_opt, comp_opt_size, "-I ./headers -DROWS=%d -DCOLS=%d -DTOTAL=%zu -DOPENCLCOMP -D%s -cl-nv-verbose", ret.g_old.param.rows, ret.g_old.param.cols, ret.g_old.param.total, integration_method);
        comp_opt[comp_opt_size - 1] = '\0';

        printf("Compile OpenCL: %s\n", comp_opt);
        cl_int err = BuildProgram(ret.gpu.program, ret.gpu.n_devs, ret.gpu.devs, comp_opt);
        BuildProgramInfo(stdout, ret.gpu.program, ret.gpu.devs[ret.gpu.i_dev], err);

        free(comp_opt);
        ret.g_old_buffer = CreateBuffer(FindGridSize(&ret.g_old), ret.gpu.ctx, CL_MEM_READ_WRITE);
        ret.g_new_buffer = CreateBuffer(FindGridSize(&ret.g_new), ret.gpu.ctx, CL_MEM_READ_WRITE);
        WriteFullGridBuffer(ret.gpu.queue, ret.g_old_buffer, &ret.g_old);
        WriteFullGridBuffer(ret.gpu.queue, ret.g_new_buffer, &ret.g_new);
        ret.gpu.kernels = InitKernels(ret.gpu.program, kernels, kernels_n);
    }

    EndParse();

#if !(defined(RK4) || defined(RK2) || defined(EULER))
    fprintf(stderr, "Invalid integration\n");
    exit(1);
#endif

    return ret;
}

void FreeSimulator(Simulator *s)
{
    if (s->grid_out_file)
    {
        free(s->grid_out_file);
        s->grid_out_file = NULL;
    }

    if (s->velxy_Ez)
    {
        free(s->velxy_Ez);
        s->velxy_Ez = NULL;
    }

    if (s->pos_xy)
    {
        free(s->pos_xy);
        s->pos_xy = NULL;
    }

    if (s->avg_mag)
    {
        free(s->avg_mag);
        s->avg_mag = NULL;
    }

    if (s->chpr_chim)
    {
        free(s->chpr_chim);
        s->chpr_chim = NULL;
    }

    FreeGrid(&s->g_old);
    FreeGrid(&s->g_new);
    if (s->use_gpu)
    {
        PrintCLError(stderr, clReleaseMemObject(s->g_old_buffer), "Error releasing g_old buffer");
        PrintCLError(stderr, clReleaseMemObject(s->g_new_buffer), "Error releasing g_new buffer");
        FreeGPU(&s->gpu);
    }
}

void ExportSimulator(Simulator *s, FILE *file)
{
    // double J_abs = s->g_old.param.exchange * (s->g_old.param.exchange < 0? -1.0: 1.0);
    double J_abs = fabs(s->g_old.param.exchange);
    fprintf(file, "Times Steps: %zu\n", s->n_steps);
    fprintf(file, "Time Step: %e\n", s->dt);
    fprintf(file, "Total Time Real: %e\n\n\n", s->dt * s->n_steps * HBAR / J_abs);

    fprintf(file, "Exchange: %e J\n", s->g_old.param.exchange);
    fprintf(file, "DMI: %e J\n", s->g_old.param.dm);
    fprintf(file, "Cubic Anisotropy: %e J\n\n", s->g_old.param.cubic_ani);
    fprintf(file, "MU_S: %e J/T\n", s->g_old.param.mu_s);
    fprintf(file, "Average Spin: %e \n", s->g_old.param.avg_spin);
    fprintf(file, "Lande: %e \n\n", s->g_old.param.lande);

    fprintf(file, "Alpha: %e \n", s->g_old.param.alpha);
    fprintf(file, "Gamma: %e \n\n", s->g_old.param.gamma);

    fprintf(file, "Integration: %s\n", integration_method);
    fprintf(file, "Write to File: %d\n", s->write_to_file);
    fprintf(file, "Write to File Cut: %zu\n", s->write_cut);
}

void ExportSimulatorFile(Simulator *s, const char *path)
{
    FILE *file = mfopen(path, "wb", 1);
    ExportSimulator(s, file);
    fclose(file);
}

void DumpWriteGrid(const char *file_path, Simulator *s)
{
    FILE *f = mfopen(file_path, "wb", 1);

    fwrite(&s->g_old.param.rows, sizeof(s->g_old.param.rows), 1, f);
    fwrite(&s->g_old.param.cols, sizeof(s->g_old.param.cols), 1, f);
    int n_steps = s->n_steps / s->write_cut;
    fwrite(&n_steps, 1, sizeof(int), f);

    size_t ss = s->write_to_file * s->n_steps * s->g_old.param.total / s->write_cut * sizeof(Vec);
    fwrite(s->grid_out_file, 1, ss, f);

    fclose(f);
}

void DumpWriteChargeGrid(const char *file_path, Simulator *s)
{
    FILE *f = mfopen(file_path, "wb", 1);

    int rows = s->g_old.param.rows;
    int cols = s->g_old.param.cols;
    fwrite(&rows, sizeof(rows), 1, f);
    fwrite(&cols, sizeof(cols), 1, f);
    int n_steps = s->n_steps / s->write_cut;
    fwrite(&n_steps, 1, sizeof(int), f);

    size_t ss = s->write_to_file * s->n_steps * s->g_old.param.total / s->write_cut * sizeof(double);

    double *charge_total = calloc(s->write_to_file * s->n_steps * s->g_old.param.total / s->write_cut, sizeof(double));
    for (int i = 0; i < n_steps; ++i)
    {
        for (size_t I = 0; I < (size_t)(rows * cols); ++I)
        {
            charge_total[i * rows * cols + I] = ChargeI(I, &s->grid_out_file[i * rows * cols], rows, cols, s->g_old.param.pbc);
        }
    }
    fwrite(charge_total, ss, 1, f);

    fclose(f);
    free(charge_total);
}
#endif
