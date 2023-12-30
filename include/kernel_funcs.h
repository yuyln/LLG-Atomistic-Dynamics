#ifndef __KERNEL_FUNCS_H
#define __KERNEL_FUNCS_H
#include "complete_kernel.h"
#include "string_view.h"

string fill_functions_on_kernel(string_view current_augment, string_view field_augment, string_view kernel_augment);
string fill_compilation_params(string_view compilation, string_view compilation_augment);
#endif
