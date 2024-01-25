#ifndef __STRING_VIEW_H
#define __STRING_VIEW_H
#include <stdarg.h>
#include <stdint.h>

#define S_FMT "%.*s"
#define S_ARG(s) (int)((s).len), (s).str

typedef struct {
    char *str;
    uint64_t len;
} string;

typedef struct {
    const char *str;
    uint64_t len;
} string_view;

void str_cat_str(string *s, string s2);
void str_cat_cstr(string *s, const char *s2);
void str_cat_fmt(string *s, const char *fmt, ...);
void str_cat_sv(string *s, string_view sv);
void str_free(string *s);
string str_from_cstr(const char *s);
string str_from_fmt(const char *fmt, ...);
const char *str_as_cstr(string *s);

const char *sv_as_cstr(string_view s);
string_view sv_from_cstr(const char *s);
#endif
