CC=gcc
CFLAGS=-std=c17 -pedantic -Wconversion -Wall -Wextra -Wshadow -Wpointer-arith -Werror -g
LIBRARIES=-lpthread

.PHONY : all
all: siggen

siggen: siggen.c
	$(CC) -o $@ $< ${LIBRARIES} ${CFLAGS}

run: siggen
	./siggen

.PHONY : clean
clean:
	rm siggen
