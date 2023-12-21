#include <stdio.h>
#include "string_view.h"
#include "grid_types.h"

int main(void) {
    pbc_rules p = 1 << 0 | 1 << 1 | 1 << 2;
    printf("%d\n", p);
    return 0;
}
