# petsc03

Educational PETSc/MPI example written by Rafel Amer (UPC) for teaching parallel programming with PETSc. This folder brings together the distributed `Vec` from [petsc01](../petsc01/README.md) and the distributed `Mat` from [petsc02](../petsc02/README.md) to solve a real linear system `K X = F` with PETSc's `KSP` linear solver framework. The code comments and the `help` string are in English, following the same conventions as petsc01/petsc02.

## Build

The `Makefile` auto-detects Ubuntu (via `lsb_release`) vs. a Fedora/RHEL-style OpenMPI installation and sets `INCLUDES`/`LIBRARIES` accordingly, exactly like petsc02's:

```
make            # builds bin/ex01 with mpicc
make clean      # removes bin/
```

- Ubuntu: `-I/usr/lib/x86_64-linux-gnu/openmpi/include/ -I/usr/lib/petscdir/petsc64-3.24/x86_64-linux-gnu-real/include/`, libs from `-L/usr/lib/x86_64-linux-gnu/`
- Fedora/RHEL: `-I/usr/include/openmpi-x86_64 -I/usr/include/openmpi-x86_64/petsc`, libs from `-L/usr/lib64/openmpi/lib`

If PETSc/MPI are installed somewhere else (for example, a PETSc build from source with `PETSC_DIR`/`PETSC_ARCH`), update `INCLUDES`/`LIBRARIES` in the Makefile instead. The compiled binary goes into `bin/`, which is covered by the repository's root `.gitignore`.

## Input data

`ex01` reads two PETSc binary files from the current directory:
- `F.dat` — the right-hand-side vector, loaded with `VecLoad`.
- `K.dat` — the sparse system matrix, loaded with `MatLoad`.

Both are symbolic links to the datasets already generated in the sibling folders (`F.dat -> ../petsc01/F.dat`, `K.dat -> ../petsc02/K.dat`), so they don't need to be regenerated here — see petsc01's `vector.py` and petsc02's `matrix.py` if the underlying `.txt` data changes.

## Running the example

`ex01` loads `F` and `K`, then solves `K X = F` with a `KSP` object configured entirely through PETSc command-line options (`KSPSetFromOptions`), so the solver/preconditioner combination can be changed without recompiling. The recommended combination is GMRES with the algebraic-multigrid preconditioner `gamg`:

```
mpirun -np 4 bin/ex01 -ksp_type gmres -pc_type gamg -ksp_rtol 1e-9
```

The program prints the number of Krylov iterations needed to reach the requested tolerance (`KSPGetIterationNumber`). This `K` is **not** symmetric positive definite (so `cg` diverges) and is poorly conditioned for a plain incomplete factorization, which makes the preconditioner choice matter a lot: `gamg` converges in about 130 iterations, whereas the single-level `-pc_type bjacobi -sub_pc_type ilu` needs several thousand for the same `-ksp_rtol 1e-9` — correct, but far slower. See the [solver table](#choosing-a-solver-example-runs) below for more combinations.

## Running in a distributed environment

As in [petsc01](../petsc01/README.md#running-in-a-distributed-environment), to run across several nodes give OpenMPI a hostfile (one `node slots=N` line per machine) and launch with `--hostfile`:

```
mpirun -np 32 --hostfile ~/hosts bin/ex01 -ksp_type gmres -pc_type gamg -ksp_rtol 1e-9
```

The same requirements apply: passwordless SSH between the nodes, the same OpenMPI/PETSc installed at the same path on every node, and the input files reachable on every node. Here `F.dat` and `K.dat` (symlinked from `../petsc01`/`../petsc02`) are loaded via MPI-IO, so they must sit on a shared filesystem or be copied to the same path on each node — following a symlink only works if its target is reachable there too. The solver table below collects exactly this kind of run on 32 MPI processes across the hosts in `~/hosts`. On a SLURM cluster (e.g. MareNostrum 5) launch the same binary with `srun` from a job script instead.

## Choosing a solver: example runs

The following table collects a set of solver/preconditioner combinations tried on a cluster (32 MPI processes across the hosts listed in `~/hosts`), useful as a starting point for experimenting with `KSP`/`PC` options on this particular system `K`:

| Command | Solver (`-ksp_type`) | Preconditioner | Notes |
|---|---|---|---|
| `-ksp_type gmres -pc_type gamg -ksp_rtol 1e-9` | GMRES | Algebraic multigrid (GAMG) | **recommended** — ~130 iterations, scalable; multigrid tames the poor conditioning that stalls single-level ILU |
| `-ksp_type gmres -pc_type bjacobi -sub_pc_type ilu -sub_pc_factor_mat_ordering_type rcm -ksp_rtol 1e-9` | GMRES | Block Jacobi, ILU per block (RCM ordering) | baseline block-Jacobi/ILU setup; correct but needs several thousand iterations on this `K` |
| same + `-ksp_gmres_restart 50` | GMRES (restart 50) | Block Jacobi, ILU (RCM) | larger GMRES restart, more memory per iteration |
| `-ksp_type bcgs -pc_type asm -pc_asm_overlap 1 -sub_pc_type ilu -sub_pc_factor_mat_ordering_type rcm -ksp_rtol 1e-9` | BiCGStab | Additive Schwarz (overlap 1), ILU (RCM) | ASM instead of block Jacobi |
| `-ksp_type tfqmr -pc_type bjacobi -sub_pc_type ilu -sub_pc_factor_mat_ordering_type np -ksp_rtol 1e-9` | TFQMR | Block Jacobi, ILU | ordering type `np` — double-check this is a valid `MatOrderingType` (e.g. `natural`) before reusing |
| same as baseline + `-sub_pc_factor_levels 2` | GMRES | Block Jacobi, ILU(2) (RCM) | higher ILU fill level |
| `-ksp_type gmres -pc_type asm -pc_asm_overlap 2 -sub_pc_factor_levels 1 -sub_pc_type ilu -sub_pc_factor_mat_ordering_type rcm -ksp_rtol 1e-9` | GMRES | Additive Schwarz (overlap 2), ILU(1) (RCM) | wider overlap, ILU(1) |
| `-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type superlu_dist` | direct (`preonly`) | Parallel LU via SuperLU_DIST | exact solve, no Krylov iterations — needs PETSc built with SuperLU_DIST |

`-ksp_type preonly` with a factorization `-pc_type` skips the iterative solve entirely and just applies the (exact) preconditioner, which is the standard PETSc idiom for a direct solve. The block/ASM preconditioners in the other rows are single-level: each MPI rank factorizes only its local block (`bjacobi`) or an overlapping subdomain (`asm`), instead of factorizing the whole matrix. Their weakness is that they do nothing to couple information across the whole domain, so on a poorly conditioned `K` the iteration count grows large (thousands here). `gamg` (the recommended row) is a *multilevel* preconditioner: it builds a hierarchy of progressively coarser problems and so keeps the iteration count low and roughly independent of the mesh size, which is why it is the better default for this system and the natural choice when scaling to more processes.

## Architecture of the example

- **ex01** — loads a distributed vector `F` (`VecLoad`) and a distributed matrix `K` (`MatLoad`) from binary files via MPI-IO, creates a `KSP` solver with `KSPCreate`/`KSPSetOperators`, lets the solver and preconditioner be fully configured from the command line with `KSPSetFromOptions`, solves `K X = F` with `KSPSolve`, and reports the iteration count with `KSPGetIterationNumber`.

When modifying or adding an example, keep the numbering scheme (`exNN.c`) and add the corresponding target to the `TARGETS` list in the Makefile.
