# petsc01

Educational PETSc/MPI examples written by Rafel Amer (UPC) for teaching parallel programming with PETSc. The code comments and the `help` strings are in English. New examples added here should follow the same style: a header block with the license (dual LGPLv3/GPLv2 license, see the existing files for the exact text), a `static char help[]` description, and `CHKERRQ(ierr)` after every PETSc call.

## Build

The `Makefile` assumes a Fedora/RHEL-style OpenMPI installation:

```
make            # builds bin/ex01 .. bin/ex06 with mpicc
make clean      # removes bin/
```

The `Makefile` has the OpenMPI/PETSc include and library paths fixed for this environment:
- Includes: `-I/usr/include/openmpi-x86_64 -I/usr/include/openmpi-x86_64/petsc`
- Libs: `-L/usr/lib64/openmpi/lib -lpetsc -lmpi`

If PETSc/MPI are installed somewhere else (for example, a PETSc build from source with `PETSC_DIR`/`PETSC_ARCH`), update `INCLUDES`/`LIBRARIES` in the Makefile instead of assuming the Fedora paths. The compiled binaries go into `bin/` and are listed in `.gitignore`.

## Running the examples

The examples are MPI programs, run with `mpirun`/`mpiexec` and a number of processes:

```
mpirun -np 4 bin/ex01
```

The behavior (ownership ranges, ghost layouts) depends on the parallel rank, so re-running with a different `-np` changes the output and is the main way to verify that an example behaves correctly across ranks.

`ex03`, `ex04` and `ex05` read `F.dat`, a PETSc binary vector loaded in parallel via `VecLoad`/MPI-IO. `F.dat` is generated from the plain-text data file `F.txt` using `vector.py` (requires `petsc4py`/`numpy`):

```
python3 vector.py   # regenerates F.dat from F.txt
```

`F.dat` only needs to be regenerated if `F.txt` changes — the binary file is stored in the repository.

## Running in a distributed environment

The `mpirun -np <N>` commands above launch all `N` processes on the local machine. To run across several nodes of a cluster, give OpenMPI a **hostfile** listing the machines and how many slots (cores) each one offers:

```
# ~/hosts
node01 slots=16
node02 slots=16
```

and launch with `--hostfile`:

```
mpirun -np 32 --hostfile ~/hosts bin/ex01
```

OpenMPI then spreads the 32 ranks over the two nodes (16 each). For a multi-node run to work:

- **Passwordless SSH** must be set up from the launch node to every host in the file — OpenMPI uses SSH to start the remote processes.
- **The same OpenMPI and PETSc** must be installed at the same path on every node (here Fedora's `/usr/lib64/openmpi`), so `mpirun`, `libpetsc` and `libmpi` resolve identically everywhere.
- **The binary and every input file must be reachable at the same path on every node** — either on a shared filesystem (NFS/Lustre, the usual HPC setup) or copied to each node. For this folder that means `bin/ex01`..`bin/ex06`, and additionally `F.dat` for `ex03`/`ex04`/`ex05`, which PETSc loads through MPI-IO and therefore must see on every node.

Passing an `-np` larger than the total number of slots oversubscribes the cores; add `--oversubscribe` if you deliberately want more ranks than slots for testing.

On a batch-scheduled cluster you do **not** call `mpirun --hostfile` yourself — the scheduler assigns the nodes. On a SLURM system such as MareNostrum 5 you launch the same binary with `srun` from inside a job script, e.g.

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=32
srun bin/ex01
```

## Architecture of the examples

The first five examples build on each other in increasing complexity, all around PETSc's distributed `Vec` object:

- **ex01** — creates a parallel `Vec`, each rank fills its own ownership range (`VecGetOwnershipRange`) with values, assembles it (`VecAssemblyBegin/End`) and displays it.
- **ex02** — parallel vector algebra: `VecScale`, `VecAXPY`, `VecDot`, `VecNorm`, with the result printed only by rank 0 via `PetscPrintf`.
- **ex03** — parallel I/O: loads a distributed vector from a PETSc binary file (`F.dat`) with `VecLoad` over MPI-IO, and prints the ownership range per rank with `PetscSynchronizedPrintf`/`PetscSynchronizedFlush`.
- **ex04** — introduces ghost vectors (`VecCreateGhost`) with a single ghost point per rank (the global index right after its local range), demonstrating halo exchange via `VecGhostUpdateBegin/End` and raw array access (`VecGetArray`) to read neighboring values.
- **ex05** — extends ex04 to a halo of 3 points on each side (`ifantasma[6]`), using `VecGhostGetLocalForm`/`VecGhostRestoreLocalForm` to obtain a local array view that includes the left and right ghost regions, with boundary checks for ranks at the global start/end.

A sixth example is independent of the previous ones and illustrates PETSc's numerical-integration machinery, the building block of the finite element assembly:

- **ex06** — computes integrals over a tetrahedron with PETSc's simplex quadrature. It builds a Stroud conical rule with `PetscDTStroudConicalQuadrature`, reads its points and weights with `PetscQuadratureGetData`, and transports the rule from PETSc's biunit reference tetrahedron to a physical one by the affine map (barycentric coordinates and the Jacobian determinant). Four polynomial integrands of increasing degree are integrated and compared against their exact values; the number of Gauss-Jacobi points per direction is set with `-qpts <n>`, and running with `-qpts 1` shows the higher-degree integrands failing once the rule's strength drops below their degree.

When modifying or adding an example, keep the numbering scheme (`exNN.c`) and add the corresponding target to the `TARGETS` list in the Makefile.
