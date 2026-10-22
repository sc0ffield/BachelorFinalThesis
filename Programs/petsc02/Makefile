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

TARGETS = bin/ex01 bin/ex02 bin/ex03

all: $(TARGETS)

bin/%: %.c
	@mkdir -p bin
	$(CC) $(CCFLAGS) -o $@ $< $(LIBRARIES) $(LIBS)

.PHONY: check
check:
	@command -v $(CC) >/dev/null 2>&1 || { echo "check: $(CC) not found"; exit 1; }
	@for src in $(TARGETS:bin/%=%.c); do \
		test -f $$src || { echo "check: missing $$src"; exit 1; }; \
		$(CC) $(INCLUDES) -fsyntax-only $$src || exit 1; \
	done
	@for geo in $(MESHES:%.msh=%.geo); do \
		test -f $$geo || { echo "check: missing $$geo"; exit 1; }; \
	done
	@for py in $(wildcard *.py); do \
		python3 -c 'import ast,sys; ast.parse(open(sys.argv[1]).read())' $$py || exit 1; \
	done
	@echo "check: OK"

clean:
	rm -f $(TARGETS)
	rmdir bin 2>/dev/null || true
