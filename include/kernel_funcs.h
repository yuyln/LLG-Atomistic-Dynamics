#ifndef __KERNEL_FUNCS_H
#define __KERNEL_FUNCS_H
#include "string_view.h"
#include "v3d.h"

string fill_functions_on_kernel(string current_augment, string field_augment, string temperature_augment, string kernel_augment);
string fill_compilation_params(string compilation, string compilation_augment);

string create_current_stt_dc(double jx, double jy, double beta);
string create_current_stt_ac(double jx, double jy, double omega, double beta);
string create_current_she_dc(double j, v3d p, double beta);
string create_current_she_ac(double j, v3d p, double omega, double beta);

string create_current_stt_dc_ac(double jx_dc, double jy_dc, double jx_ac, double jy_ac, double omega, double beta);
string create_current_she_dc_ac(double j_dc, v3d p_dc, double j_ac, v3d p_ac, double omega, double beta);

string create_field_tesla(v3d field_tesla);
string create_field_D2_over_J(v3d field_D2_over_J, double J, double D, double mu);
string create_field_J(v3d field_J, double J, double mu);
string create_temperature(double temperature);

#endif
