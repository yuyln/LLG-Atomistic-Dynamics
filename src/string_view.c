#include "string_view.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

string string_init(uint64_t len) {
    return (string){.str = calloc(len, 1),
                    .len = len};
}

uint64_t string_clear(string *s) {
    return s->len = 0;
}

void string_free(string *s) {
    free(s->str);
    memset(s, 0, sizeof(string));
}

const char *string_as_cstr(string *s) {
    char *new = realloc(s->str, s->len + 1);
    if (!new) {
        fprintf(stderr, "[ FATAL ] Could not realloc string \"%*.s\" to null termianted string", (int)s->len, s->str);
        exit(1);
    }
    s->str = new;
    s->str[s->len] = '\0';
    s->len++;
    return s->str;
}

string string_from_cstr(const char *str) {
    string ret = string_init(strlen(str));
    memcpy(ret.str, str, ret.len);
    return ret;
}

string_view sv_from_string(string s, uint64_t start, uint64_t end) {
    if (start >= s.len)
        start = s.len - 1;
    if (end >= s.len)
        end = s.len - 1;
    return (string_view){.str = &s.str[start], .len = end - start};
}
