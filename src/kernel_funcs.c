#include "kernel_funcs.h"

string fill_functions_on_kernel(string_view current_augment, string_view field_augment, string_view temperature_augment, string_view kernel_augment) {
    string ret = {0};
    str_cat_cstr(&ret, complete_kernel);

    str_cat_cstr(&ret, "\n\n\n\n");
    str_cat_cstr(&ret, "current generate_current(grid_site_param gs, double time) {\n");

    str_cat_sv(&ret, current_augment);
    str_cat_cstr(&ret, "\n}\n");

    str_cat_cstr(&ret, "\n\n\n\n");
    str_cat_cstr(&ret, "v3d generate_magnetic_field(grid_site_param gs, double time) {\n");

    str_cat_sv(&ret, field_augment);
    str_cat_cstr(&ret, "\n}\n");

    str_cat_cstr(&ret, "\n\n\n\n");
    str_cat_cstr(&ret, "double generate_temperature(grid_site_param gs, double time) {\n");

    str_cat_sv(&ret, temperature_augment);
    str_cat_cstr(&ret, "\n}\n");

    str_cat_sv(&ret, kernel_augment);

    return ret;
}

string fill_compilation_params(string_view compilation, string_view compilation_augment) {
    string ret = {0};
    str_cat_sv(&ret, compilation);
    str_cat_cstr(&ret, " ");
    str_cat_sv(&ret, compilation_augment);
    return ret;
}
