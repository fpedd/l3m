# A CFLAGS on the command line replaces this default; the lines below are always added.
CFLAGS  ?= -O3 -march=x86-64-v3 -Wall -Wextra -fno-math-errno -ffp-contract=fast
override CFLAGS += -std=c11 -fPIC -fvisibility=hidden -D_GNU_SOURCE
LDLIBS   = -lpthread -lm

LIB_SRC := $(filter-out src/main.c, $(wildcard src/*.c))
LIB_OBJ := $(LIB_SRC:src/%.c=build/%.o)

build/kernels_avx2.o:   override CFLAGS += -march=x86-64-v3
build/kernels_avx512.o: override CFLAGS += -march=x86-64-v4 -mavx512vnni -mavx512bf16 -mprefer-vector-width=512

all: l3m libl3m.so

libl3m.so: $(LIB_OBJ)
	$(CC) -shared -o $@ $^ $(LDLIBS)

l3m: src/main.c libl3m.so
	$(CC) $(CFLAGS) -o $@ $< -L. -ll3m -Wl,-rpath,'$$ORIGIN' $(LDLIBS)

build/%.o: src/%.c src/*.h | build
	$(CC) $(CFLAGS) -c -o $@ $<

build:
	mkdir -p build

# Tests and benchmarks link the objects directly, for the non-exported symbols.
tests/test_kernels: tests/test_kernels.c $(LIB_OBJ)
	$(CC) $(CFLAGS) -Isrc -o $@ $^ $(LDLIBS)

build/%: bench/%.c $(LIB_OBJ) | build
	$(CC) $(CFLAGS) -Isrc -o $@ $^ $(LDLIBS)

test: tests/test_kernels
	./tests/test_kernels

bench: $(patsubst bench/%.c, build/%, $(wildcard bench/*.c))

clean:
	rm -rf build l3m libl3m.so tests/test_kernels

.PHONY: all test bench clean
