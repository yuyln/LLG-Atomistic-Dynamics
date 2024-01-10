#include "kernel_funcs.h"

string fill_functions_on_kernel(string_view current_augment, string_view field_augment, string_view temperature_augment, string_view kernel_augment) {
    string ret = {0};
    string_add_cstr(&ret, complete_kernel);

    string_add_cstr(&ret, "\n\n\n\n");
    string_add_cstr(&ret, "current generate_current(grid_site_param gs, double time) {\n");

    string_add_sv(&ret, current_augment);
    string_add_cstr(&ret, "\n}\n");

    string_add_cstr(&ret, "\n\n\n\n");
    string_add_cstr(&ret, "v3d generate_magnetic_field(grid_site_param gs, double time) {\n");

    string_add_sv(&ret, field_augment);
    string_add_cstr(&ret, "\n}\n");

    string_add_cstr(&ret, "\n\n\n\n");
    string_add_cstr(&ret, "double generate_temperature(grid_site_param gs, double time) {\n");

    string_add_sv(&ret, temperature_augment);
    string_add_cstr(&ret, "\n}\n");

    string_add_sv(&ret, kernel_augment);

    return ret;
}

string fill_compilation_params(string_view compilation, string_view compilation_augment) {
    string ret = {0};
    string_add_sv(&ret, compilation);
    string_add_cstr(&ret, " ");
    string_add_sv(&ret, compilation_augment);
    return ret;
}
