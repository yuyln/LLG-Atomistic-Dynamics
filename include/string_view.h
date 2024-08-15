#ifndef __string_H
#define __string_H
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>

#define S_FMT "%.*s"
#define S_ARG(s) (int)((s).len), (s).str

typedef struct {
    char *items;
    uint64_t len;
    uint64_t cap;
} string_builder;

void sb_cat_sb(string_builder *s, string_builder s2);
void sb_cat_cstr(string_builder *s, const char *s2);
void sb_cat_fmt(string_builder *s, const char *fmt, ...);
void sb_free(string_builder *s);
char *sb_as_cstr(string_builder *s);
#endif
