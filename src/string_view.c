#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "string_view.h"

void string_add_cstr(string *s, const char *str) {
    uint64_t old_len = s->len;
    s->len += strlen(str);
    s->str = realloc(s->str, s->len);
    memcpy(&s->str[old_len], str, strlen(str));
}

void string_add_sv(string *s, string_view sv) {
    uint64_t old_len = s->len;
    s->str = realloc(s->str, old_len + sv.len);
    s->len += sv.len;
    memcpy(&s->str[old_len], sv.str, sv.len);
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

string_view sv_from_string(string s, uint64_t start, uint64_t end) {
    if (start >= s.len)
        start = s.len - 1;
    if (end >= s.len)
        end = s.len - 1;
    return (string_view){.str = &s.str[start], .len = end - start};
}

string_view sv_from_cstr(const char *str) {
    return (string_view){.str = (char *const) str, .len = strlen(str)};
}
