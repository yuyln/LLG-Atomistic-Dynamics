#ifndef __STRING_VIEW_H
#define __STRING_VIEW_H
#include <stdint.h>

typedef struct {
    const char *str;
    uint64_t len;
} string_view;

typedef struct {
    char *str;
    uint64_t len;
} string;

void string_add_cstr(string *s, const char *str);
void string_free(string *s);
const char *string_as_cstr(string *s);
void string_add_sv(string *s, string_view sv);

string_view sv_from_string(string s, uint64_t start, uint64_t end);
string_view sv_from_cstr(const char *str);

//string_view sv_next_token(string_view s, const char *sep);

#endif
