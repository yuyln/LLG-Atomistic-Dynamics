#ifndef __string_H
#define __string_H
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>

#define S_FMT "%.*s"
#define S_ARG(s) (int)((s).len), (s).str
#define str_is_cstr(s) (string){.str=(char*)(s), .len=strlen((s)), .can_manipulate=false}
#define STR_NULL (string){.str="\0", .len=0, .can_manipulate=false}

typedef struct {
    char *str;
    uint64_t len;
    bool can_manipulate;
} string;

void str_cat_str(string *s, string s2);
void str_cat_cstr(string *s, const char *s2);
void str_cat_fmt(string *s, const char *fmt, ...);
void str_free(string *s);
string str_from_cstr(const char *s);
string str_from_fmt(const char *fmt, ...);
const char *str_as_cstr(string *s);
#endif
