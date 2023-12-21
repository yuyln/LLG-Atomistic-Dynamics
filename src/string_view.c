#include "string_view.h"
#include <stdlib.h>
#include <string.h>

string string_init(uint64_t len) {
    return (string){.str = calloc(len, 1),
                    .len = len,
                    .cap = len};
}

uint64_t string_clear(string *s) {
    return s->len = 0;
}

void string_free(string *s) {
    free(s->str);
    memset(s, 0, sizeof(string));
}

string_view sv_from_string(string s, uint64_t start, uint64_t end) {
    if (start >= s.len)
        start = s.len - 1;
    if (end >= s.len)
        end = s.len - 1;
    return (string_view){.str = &s.str[start], .len = end - start};
}
