# https://stackoverflow.com/a/8946900/7829525
C99_STRICT := -Wall -Wextra -std=c99 -pedantic -Wmissing-prototypes -Wstrict-prototypes \
    -Wold-style-definition

bin-anon_struct: anon_struct.c
	$(CC) $(C99_STRICT) $< -o $@
