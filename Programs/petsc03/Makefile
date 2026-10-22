OS := $(shell lsb_release -si)
ifeq ($(OS),Ubuntu)
	INCLUDES = -I. -I/usr/lib/x86_64-linux-gnu/openmpi/include/ -I/usr/lib/petscdir/petsc64-3.24/x86_64-linux-gnu-real/include/
	LIBRARIES = -L/usr/lib/x86_64-linux-gnu/
else
	INCLUDES = -I. -I/usr/include/openmpi-x86_64 -I/usr/include/openmpi-x86_64/petsc
	LIBRARIES = -L/usr/lib64/openmpi/lib
endif

CCFLAGS = -O3 $(INCLUDES) $(LIBRARIES)
LIBS = -lpetsc -lmpi
CC = mpicc

TARGETS = bin/ex01

all: $(TARGETS)

bin/%: %.c
	@mkdir -p bin
	$(CC) $(CCFLAGS) -o $@ $< $(LIBRARIES) $(LIBS)

clean:
	rm -f $(TARGETS)
	rmdir bin 2>/dev/null || true
