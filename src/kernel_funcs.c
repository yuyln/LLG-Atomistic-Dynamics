#include "kernel_funcs.h"
#include "complete_kernel.h"

string fill_functions_on_kernel(string current_augment, string field_augment, string temperature_augment, string kernel_augment) {
    string ret = str_from_cstr("");
    str_cat_cstr(&ret, complete_kernel);

    str_cat_cstr(&ret, "\n\n\n\n");
    str_cat_cstr(&ret, "current generate_current(grid_site_params gs, double time) {\n");

    str_cat_str(&ret, current_augment);
    str_cat_cstr(&ret, "\n}\n");

    str_cat_cstr(&ret, "\n\n\n\n");
    str_cat_cstr(&ret, "v3d generate_magnetic_field(grid_site_params gs, double time) {\n");

    str_cat_str(&ret, field_augment);
    str_cat_cstr(&ret, "\n}\n");

    str_cat_cstr(&ret, "\n\n\n\n");
    str_cat_cstr(&ret, "double generate_temperature(grid_site_params gs, double time) {\n");

    str_cat_str(&ret, temperature_augment);
    str_cat_cstr(&ret, "\n}\n");

    str_cat_str(&ret, kernel_augment);

    return ret;
}

string fill_compilation_params(string compilation, string compilation_augment) {
    string ret = str_from_cstr("");
    str_cat_str(&ret, compilation);
    str_cat_cstr(&ret, " ");
    str_cat_str(&ret, compilation_augment);
    return ret;
}
