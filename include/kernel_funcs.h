#ifndef __KERNEL_FUNCS_H
#define __KERNEL_FUNCS_H
#include "v3d.h"

char *fill_functions_on_kernel(const char *current_augment, const char *field_augment, const char *temperature_augment, const char *kernel_augment);
char *fill_compilation_params(const char *compilation, const char *compilation_augment);

char *create_current_stt_dc(double jx, double jy, double beta);
char *create_current_stt_ac(double jx, double jy, double omega, double beta);
char *create_current_she_dc(double j, v3d p, double beta);
char *create_current_she_ac(double j, v3d p, double omega, double beta);

char *create_current_stt_dc_ac(double jx_dc, double jy_dc, double jx_ac, double jy_ac, double omega, double beta);
char *create_current_she_dc_ac(double j_dc, v3d p_dc, double j_ac, v3d p_ac, double omega, double beta);

char *create_field_tesla(v3d field_tesla);
char *create_field_D2_over_J(v3d field_D2_over_J, double J, double D, double mu);
char *create_field_J(v3d field_J, double J, double mu);
char *create_temperature(double temperature);

#endif
