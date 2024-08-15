#include "kernel_funcs.h"
#include "complete_kernel.h"
#include "constants.h"
#include "string_builder.h"
#include "logging.h"
#include <stdio.h>

char *fill_functions_on_kernel(const char *current_augment, const char *field_augment, const char *temperature_augment, const char *kernel_augment) {
    string_builder ret = {0};
    sb_cat_cstr(&ret, complete_kernel);

    sb_cat_cstr(&ret, "\n\n\n\n");
    sb_cat_cstr(&ret, "current generate_current(grid_site_params gs, double time) {\n");

    if (current_augment)
        sb_cat_cstr(&ret, current_augment);
    sb_cat_cstr(&ret, "\n}\n");

    sb_cat_cstr(&ret, "\n\n\n\n");
    sb_cat_cstr(&ret, "v3d generate_magnetic_field(grid_site_params gs, double time) {\n");

    if (field_augment)
        sb_cat_cstr(&ret, field_augment);
    sb_cat_cstr(&ret, "\n}\n");

    sb_cat_cstr(&ret, "\n\n\n\n");
    sb_cat_cstr(&ret, "double generate_temperature(grid_site_params gs, double time) {\n");

    if (temperature_augment)
        sb_cat_cstr(&ret, temperature_augment);
    sb_cat_cstr(&ret, "\n}\n");

    if (kernel_augment)
        sb_cat_cstr(&ret, kernel_augment);

    return sb_as_cstr(&ret);
}

char *fill_compilation_params(const char *compilation, const char *compilation_augment) {
    string_builder ret = {0};
    if (compilation)
        sb_cat_cstr(&ret, compilation);

    sb_cat_cstr(&ret, " ");

    if (compilation_augment)
        sb_cat_cstr(&ret, compilation_augment);
    return sb_as_cstr(&ret);
}

char *create_current_stt_dc(double jx, double jy, double beta) {
    static char buffer[2048];
    int written = snprintf(buffer, sizeof(buffer) - 1, "return (current){.type = CUR_STT, .stt.j = v3d_c(%.15e, %.15e, 0), .stt.beta = %.15e, .stt.polarization = -1};", jx, jy, beta);
    buffer[sizeof(buffer) - 1] = '\0';

    if (written < 0)
        logging_log(LOG_FATAL, "Something went wrong during `snprintf`. This should never happen, there is something very wrong with your machine");

    if (written == 0)
        logging_log(LOG_WARNING, "Not a single byte wrote during `snprintf`. Something went wrong");

    if (written == (sizeof(buffer) - 1))
        logging_log(LOG_WARNING, "During `snprintf` all bytes in the buffer were used. This probably indicates that something will go wrong in the simulation. Be careful");

    return buffer;
}

char *create_current_stt_ac(double jx, double jy, double omega, double beta) {
    static char buffer[2048];
    int written = snprintf(buffer, sizeof(buffer) - 1, "return (current){.type = CUR_STT, .stt.j = v3d_c(%.15e * cos(2.0 * M_PI * %.15e * time), %.15e * sin(2.0 * M_PI * %.15e * time), 0), .stt.beta = %.15e, .stt.polarization = -1};", jx, omega, jy, omega, beta);

    if (written < 0)
        logging_log(LOG_FATAL, "Something went wrong during `snprintf`. This should never happen, there is something very wrong with your machine");

    if (written == 0)
        logging_log(LOG_WARNING, "Not a single byte wrote during `snprintf`. Something went wrong");

    if (written == (sizeof(buffer) - 1))
        logging_log(LOG_WARNING, "During `snprintf` all bytes in the buffer were used. This probably indicates that something will go wrong in the simulation. Be careful");

    return buffer;
}

char *create_current_she_dc(double j, v3d p, double beta) {
    static char buffer[2048];
    p = v3d_normalize(p);
    p = v3d_scalar(p, j);
    int written = snprintf(buffer, sizeof(buffer) - 1, "return (current){.type = CUR_SHE, .she.p = v3d_c(%.15e, %.15e, %.15e), .she.theta_sh = -1, .she.thickness = gs.lattice, .she.beta= %.15e};", p.x, p.y, p.z, beta);

    if (written < 0)
        logging_log(LOG_FATAL, "Something went wrong during `snprintf`. This should never happen, there is something very wrong with your machine");

    if (written == 0)
        logging_log(LOG_WARNING, "Not a single byte wrote during `snprintf`. Something went wrong");

    if (written == (sizeof(buffer) - 1))
        logging_log(LOG_WARNING, "During `snprintf` all bytes in the buffer were used. This probably indicates that something will go wrong in the simulation. Be careful");

    return buffer;
}

char *create_current_she_ac(double j, v3d p, double omega, double beta) {
    static char buffer[2048];
    p = v3d_normalize(p);
    p = v3d_scalar(p, j);
    int written = snprintf(buffer, sizeof(buffer) - 1, "return (current){.type = CUR_SHE, .she.p = v3d_c(%.15e * cos(2.0 * M_PI * %.15e * time), %.15e * sin(2.0 * M_PI * %.15e * time), %.15e * sin(2.0 * M_PI * %.15e * time)), .she.theta_sh = -1, .she.thickness = gs.lattice, .she.beta= %.15e};", p.x, omega, p.y, omega, p.z, omega, beta);

    if (written < 0)
        logging_log(LOG_FATAL, "Something went wrong during `snprintf`. This should never happen, there is something very wrong with your machine");

    if (written == 0)
        logging_log(LOG_WARNING, "Not a single byte wrote during `snprintf`. Something went wrong");

    if (written == (sizeof(buffer) - 1))
        logging_log(LOG_WARNING, "During `snprintf` all bytes in the buffer were used. This probably indicates that something will go wrong in the simulation. Be careful");


    return buffer;
}

char *create_current_stt_dc_ac(double jx_dc, double jy_dc, double jx_ac, double jy_ac, double omega, double beta) {
    static char buffer[2048];

    int written = snprintf(buffer, sizeof(buffer) - 1, "return (current){.type = CUR_STT, .stt.j = v3d_c(%.15e + %.15e * cos(2.0 * M_PI * %.15e * time), %.15e + %.15e * sin(2.0 * M_PI * %.15e * time), 0), .stt.beta = %.15e, .stt.polarization = -1};", jx_dc, jx_ac, omega, jy_dc, jy_ac, omega, beta);

    if (written < 0)
        logging_log(LOG_FATAL, "Something went wrong during `snprintf`. This should never happen, there is something very wrong with your machine");

    if (written == 0)
        logging_log(LOG_WARNING, "Not a single byte wrote during `snprintf`. Something went wrong");

    if (written == (sizeof(buffer) - 1))
        logging_log(LOG_WARNING, "During `snprintf` all bytes in the buffer were used. This probably indicates that something will go wrong in the simulation. Be careful");


    return buffer;
}

char *create_current_she_dc_ac(double j_dc, v3d p_dc, double j_ac, v3d p_ac, double omega, double beta) {
    static char buffer[2048];

    p_dc = v3d_normalize(p_dc);
    p_dc = v3d_scalar(p_dc, j_dc);

    p_ac = v3d_normalize(p_ac);
    p_ac = v3d_scalar(p_ac, j_ac);

    int written = snprintf(buffer, sizeof(buffer) - 1, "return (current){.type = CUR_SHE, .she.p = v3d_c(%.15e + %.15e * cos(2.0 * M_PI * %.15e * time), %.15e + %.15e * sin(2.0 * M_PI * %.15e * time), %.15e + %.15e * sin(2.0 * M_PI * %.15e * time)), .she.theta_sh = -1, .she.thickness = gs.lattice, .she.beta= %.15e};", p_dc.x, p_ac.x, omega, p_dc.y, p_ac.y, omega, p_dc.z, p_ac.z, omega, beta);

    if (written < 0)
        logging_log(LOG_FATAL, "Something went wrong during `snprintf`. This should never happen, there is something very wrong with your machine");

    if (written == 0)
        logging_log(LOG_WARNING, "Not a single byte wrote during `snprintf`. Something went wrong");

    if (written == (sizeof(buffer) - 1))
        logging_log(LOG_WARNING, "During `snprintf` all bytes in the buffer were used. This probably indicates that something will go wrong in the simulation. Be careful");


    return buffer;
}

char *create_field_tesla(v3d field) {
    static char buffer[2048];
    int written = snprintf(buffer, sizeof(buffer) - 1, "return v3d_c(%.15e, %.15e, %.15e);", field.x, field.y, field.z);

    if (written < 0)
        logging_log(LOG_FATAL, "Something went wrong during `snprintf`. This should never happen, there is something very wrong with your machine");

    if (written == 0)
        logging_log(LOG_WARNING, "Not a single byte wrote during `snprintf`. Something went wrong");

    if (written == (sizeof(buffer) - 1))
        logging_log(LOG_WARNING, "During `snprintf` all bytes in the buffer were used. This probably indicates that something will go wrong in the simulation. Be careful");

    return buffer;
}

char *create_field_D2_over_J(v3d field, double J, double D, double mu) {
    static char buffer[2048];
    D *= SIGN(D);
    J *= SIGN(J);
    mu *= SIGN(mu);
    int written = snprintf(buffer, sizeof(buffer) - 1, "return v3d_c(%.15e, %.15e, %.15e);", field.x * D * D / J * 1.0 / mu, field.y * D * D / J * 1.0 / mu, field.z * D * D / J * 1.0 / mu);

    if (written < 0)
        logging_log(LOG_FATAL, "Something went wrong during `snprintf`. This should never happen, there is something very wrong with your machine");

    if (written == 0)
        logging_log(LOG_WARNING, "Not a single byte wrote during `snprintf`. Something went wrong");

    if (written == (sizeof(buffer) - 1))
        logging_log(LOG_WARNING, "During `snprintf` all bytes in the buffer were used. This probably indicates that something will go wrong in the simulation. Be careful");

    return buffer;
}

char *create_field_J(v3d field, double J, double mu) {
    static char buffer[2048];
    J *= SIGN(J);
    mu *= SIGN(mu);
    int written = snprintf(buffer, sizeof(buffer) - 1, "return v3d_c(%.15e, %.15e, %.15e);", field.x * J * 1.0 / mu, field.y * J * 1.0 / mu, field.z * J * 1.0 / mu);

    if (written < 0)
        logging_log(LOG_FATAL, "Something went wrong during `snprintf`. This should never happen, there is something very wrong with your machine");

    if (written == 0)
        logging_log(LOG_WARNING, "Not a single byte wrote during `snprintf`. Something went wrong");

    if (written == (sizeof(buffer) - 1))
        logging_log(LOG_WARNING, "During `snprintf` all bytes in the buffer were used. This probably indicates that something will go wrong in the simulation. Be careful");

    return buffer;
}

char *create_temperature(double temperature) {
    static char buffer[2048];
    temperature *= SIGN(temperature);
    int written = snprintf(buffer, sizeof(buffer) - 1, "return %.15e;", temperature);

    if (written < 0)
        logging_log(LOG_FATAL, "Something went wrong during `snprintf`. This should never happen, there is something very wrong with your machine");

    if (written == 0)
        logging_log(LOG_WARNING, "Not a single byte wrote during `snprintf`. Something went wrong");

    if (written == (sizeof(buffer) - 1))
        logging_log(LOG_WARNING, "During `snprintf` all bytes in the buffer were used. This probably indicates that something will go wrong in the simulation. Be careful");

    return buffer;
}
