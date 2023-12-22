#ifndef __STRING_VIEW_H
#define __STRING_VIEW_H
#include <stdint.h>

typedef struct {
    char * const str;
    uint64_t len;
} string_view;

typedef struct {
    char *str;
    uint64_t len;
} string;

string string_init(uint64_t len);
uint64_t string_clear(string *s);
void string_free(string *s);
const char *string_as_cstr(string *s);
string string_from_cstr(const char *str);

string_view sv_from_string(string s, uint64_t start, uint64_t end);

//string_view sv_next_token(string_view s, const char *sep);

#endif
