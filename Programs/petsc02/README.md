# petsc02

Educational PETSc/MPI examples written by Rafel Amer (UPC) for teaching parallel programming with PETSc. This folder builds on [petsc01](../petsc01/README.md) by moving from distributed `Vec` objects to distributed sparse `Mat` objects. The code comments and the `help` strings are in English. New examples added here should follow the same style: a header block with the license (dual LGPLv3/GPLv2 license, see the existing files for the exact text), a `static char help[]` description, and `CHKERRQ(ierr)` after every PETSc call.

## Build

The `Makefile` auto-detects Ubuntu (via `lsb_release`) vs. a Fedora/RHEL-style OpenMPI installation and sets `INCLUDES`/`LIBRARIES` accordingly:

```
make            # builds bin/ex01 .. bin/ex03 with mpicc
make clean      # removes bin/
```

- Ubuntu: `-I/usr/lib/x86_64-linux-gnu/openmpi/include/ -I/usr/lib/petscdir/petsc64-3.24/x86_64-linux-gnu-real/include/`, libs from `-L/usr/lib/x86_64-linux-gnu/`
- Fedora/RHEL: `-I/usr/include/openmpi-x86_64 -I/usr/include/openmpi-x86_64/petsc`, libs from `-L/usr/lib64/openmpi/lib`

If PETSc/MPI are installed somewhere else (for example, a PETSc build from source with `PETSC_DIR`/`PETSC_ARCH`), update `INCLUDES`/`LIBRARIES` in the Makefile instead. The compiled binaries go into `bin/`, which is covered by the repository's root `.gitignore`.

## Running the examples

The examples are MPI programs, run with `mpirun`/`mpiexec` and a number of processes:

```
mpirun -np 4 bin/ex01
```

`ex01` and `ex02` accept a `-n <size>` option to change the global size of the tridiagonal matrix (and, for `ex02`, of the vectors) without recompiling, e.g. `mpirun -np 4 bin/ex01 -n 100`.

`ex03` reads `K.dat` by default, a PETSc binary matrix loaded in parallel via `MatLoad`/MPI-IO; pass `-f <file>` to load a different binary matrix file, e.g. `mpirun -np 4 bin/ex03 -f other.dat`. `K.dat` (plus its companion `K.dat.info`) is generated from the plain-text sparse triplet file `K.txt` (`row col value` per line, 0-based indices) using `matrix.py` (requires `petsc4py`, `numpy` and `scipy`):

```
python3 matrix.py   # regenerates K.dat from K.txt
```

`K.dat` only needs to be regenerated if `K.txt` changes — the binary file is stored in the repository.

## Running in a distributed environment

As in [petsc01](../petsc01/README.md#running-in-a-distributed-environment), to run across several nodes give OpenMPI a hostfile (one `node slots=N` line per machine) and launch with `--hostfile`:

```
mpirun -np 32 --hostfile ~/hosts bin/ex01 -n 100
```

The same requirements apply: passwordless SSH between the nodes, the same OpenMPI/PETSc installed at the same path on every node, and every input file reachable on every node. Here `ex03` reads `K.dat` (and its companion `K.dat.info`) via MPI-IO, so both must live on a shared filesystem or be copied to the same path on each node; `ex01`/`ex02` build their matrix in memory and need no data files. On a SLURM cluster (e.g. MareNostrum 5) launch the same binary with `srun` from a job script instead of calling `mpirun --hostfile` directly.

## Architecture of the examples

The three examples build on each other in increasing complexity, all around PETSc's distributed `Mat` object:

- **ex01** — creates a distributed sparse tridiagonal `Mat` (`MATAIJ`, 10x10) by hand: each rank fills its own ownership range (`MatGetOwnershipRange`) with `MatSetValues`, preallocates with `MatSeqAIJSetPreallocation`/`MatMPIAIJSetPreallocation`, assembles it (`MatAssemblyBegin/End`) and displays it with `MatView`.
- **ex02** — builds the same tridiagonal `Mat` and multiplies it by a vector of ones (`MatMult`, `y = A * x`), using `MatCreateVecs` so that `x`/`y` share the matrix's row distribution, and views the result in MATLAB ASCII format.
- **ex03** — parallel I/O: loads a distributed sparse matrix from a PETSc binary file (`K.dat`) with `MatLoad` over MPI-IO, and prints the ownership range per rank with `PetscSynchronizedPrintf`/`PetscSynchronizedFlush` before displaying the full matrix with `MatView`.

When modifying or adding an example, keep the numbering scheme (`exNN.c`) and add the corresponding target to the `TARGETS` list in the Makefile.
