# petsc04

Educational PETSc/MPI examples written by Rafel Amer (UPC) for teaching parallel programming with PETSc. This folder moves from the algebraic objects (`Vec`/`Mat`) of [petsc01](../petsc01/README.md)/[petsc02](../petsc02/README.md)/[petsc03](../petsc03/README.md) to unstructured-mesh handling with `DMPlex`: reading a Gmsh mesh, partitioning and distributing it in parallel with METIS, inspecting boundary labels, and calling into embedded Lua scripts to evaluate per-point or per-process functions. The code comments and the `help` strings are in English, following the same conventions as the previous folders.

## Build

The `Makefile` auto-detects Ubuntu (via `lsb_release`) vs. a Fedora/RHEL-style OpenMPI installation and sets `INCLUDES`/`LIBRARIES` accordingly, exactly like petsc02/petsc03's, and additionally links against Lua:

```
make            # builds bin/ex01 .. bin/ex07 with mpicc
make clean      # removes bin/
```

- Ubuntu: `-I/usr/lib/x86_64-linux-gnu/openmpi/include/ -I/usr/lib/petscdir/petsc64-3.24/x86_64-linux-gnu-real/include/ -I/usr/include/lua5.3`, libs from `-L/usr/lib/x86_64-linux-gnu/` plus `-llua5.3`
- Fedora/RHEL: `-I/usr/include/openmpi-x86_64 -I/usr/include/openmpi-x86_64/petsc`, libs from `-L/usr/lib64/openmpi/lib` plus `-llua`

If PETSc/MPI/Lua are installed somewhere else (for example, a PETSc build from source with `PETSC_DIR`/`PETSC_ARCH`, or a different Lua version on Ubuntu), update `INCLUDES`/`LIBRARIES`/`LUALIBS` in the Makefile instead. The compiled binaries go into `bin/`, which is covered by the repository's root `.gitignore`.

## Input data

The first five examples call `DMPlexCreateGmshFromFile()`, default to `torus.msh`, and expect a Gmsh mesh in legacy ASCII format (v2), read as `-mesh_input <file>`. `ex06` needs no mesh: it reads a single tetrahedron and an integrand from a Lua file. `ex07` also reads a Gmsh mesh with `-mesh_input <file>`, but defaults to the Fichera-corner mesh `../petsc07/cube3.msh` and, unlike the others, *rewrites* it by metric-based adaptive refinement (see below).

`torus.msh` is **not** included in the repository, but it can be generated from [torus.geo](torus.geo) with the Makefile's `%.msh: %.geo` pattern rule, which runs Gmsh in 3D with 32 threads and writes legacy ASCII (v2) format:

```
make torus.msh   # runs: gmsh torus.geo -3 -nt 32 -format msh2 -o torus.msh
```

Once the mesh is available:

```
mpirun -np 4 bin/ex01 -mesh_input torus.msh
```

`ex03`, `ex04` and `ex05` also read a Lua file with `-lua_input <file>`:
- `ex03` always loads `functions.lua`, which defines `compute_boundary(rank, base)` and `compute_interior(rank, base)`.
- `ex04` and `ex05` default to `function.lua`, which defines `a`, `b1`, `b2`, `b3`, `c`, `f` (FEM-style coefficients as functions of a point `(x, y, z)`); pass `-lua_input functions.lua` to evaluate a different set of global functions instead (any name not found there is simply skipped, defaulting to `0.0`).
- `ex06` defaults to `tetrahedron.lua`, which defines a table `vertices` (the four vertices of a tetrahedron), a function `f(x, y, z)` (the integrand) and an optional `qpts` (quadrature points per direction).

## Running the examples

```
mpirun -np 4 bin/ex01 -mesh_input torus.msh
mpirun -np 4 bin/ex02 -mesh_input torus.msh
mpirun -np 4 bin/ex03 -mesh_input torus.msh
mpirun -np 4 bin/ex04 -mesh_input torus.msh
mpirun -np 4 bin/ex05 -mesh_input torus.msh
mpirun -np 1 bin/ex06 -lua_input tetrahedron.lua
mpirun -np 1 bin/ex07 -dm_adaptor mmg       # serial adaptive refinement (Mmg)
mpirun -np 4 bin/ex07 -dm_adaptor parmmg    # parallel adaptive refinement (ParMmg)
```

## Running in a distributed environment

As in [petsc01](../petsc01/README.md#running-in-a-distributed-environment), to run across several nodes give OpenMPI a hostfile (one `node slots=N` line per machine) and launch with `--hostfile`:

```
mpirun -np 32 --hostfile ~/hosts bin/ex01 -mesh_input torus.msh
```

The same requirements apply: passwordless SSH between the nodes, the same OpenMPI/PETSc/Lua installed at the same path on every node, and every input file reachable on every node. PETSc reads the mesh (`torus.msh`) and then `DMPlexDistribute` scatters the partitions across ranks, while each rank opens its own copy of the Lua script (`functions.lua`, `function.lua` or `tetrahedron.lua`), so both the mesh and the Lua files must live on a shared filesystem or be copied to the same path on each node. The heavier the mesh, the more distributing it across nodes pays off — a torus of a few thousand tetrahedra is trivial, but the same command scales to the multi-million-tetrahedron meshes of the later folders. On a SLURM cluster (e.g. MareNostrum 5) launch the same binary with `srun` from a job script instead.

## Architecture of the examples

- **ex01** — reads a Gmsh mesh with `DMPlexCreateGmshFromFile`, partitions and redistributes it in parallel with `DMPlexDistribute` (METIS), counts the local cells (tetrahedra) owned by each process with `DMPlexGetHeightStratum`, and prints a summary of the distributed `DM` with `DMView`.
- **ex02** — builds on ex01: it also lets command-line options configure the `DM` (`DMSetFromOptions`), imports Gmsh physical groups as the `Face Sets` label (falling back to the native `marker` label if absent), and counts the local boundary faces whose label value is set, reporting them per process with `PetscSynchronizedPrintf`/`PetscSynchronizedFlush`.
- **ex03** — same mesh/boundary-label handling as ex02, plus an embedded Lua interpreter (`lua.h`/`lualib.h`/`lauxlib.h`): each process starts its own `lua_State`, loads `functions.lua`, and calls `compute_boundary(rank, base)`/`compute_interior(rank, base)` with a per-rank base factor, reporting the results together with the local boundary-face count.
- **ex04** — same mesh/boundary-label handling as ex02/ex03, but takes the Lua file to load from `-lua_input` and iterates over a fixed list of function names (`a`, `b1`, `b2`, `b3`, `c`, `f`), calling whichever of them actually exist as global Lua functions in that file at a fixed point `(x, y, z)`, and reports all six values per process together with the local boundary-face count.
- **ex05** — evaluates the same kind of Lua functions as ex04, but at real mesh locations instead of a single fixed point: for every local cell it computes the cell centroid with `DMPlexComputeCellGeometryFVM` (which requires the mesh to be interpolated, `PETSC_TRUE`), evaluates `a`, `b1`, `b2`, `b3`, `c`, `f` there, and reports the per-process average of each function over all of its local cell centroids.
- **ex06** — computes the integral of a function over a single tetrahedron, with **both** the tetrahedron and the integrand defined in the Lua file (no mesh involved). It reads the four vertices from the Lua table `vertices` and the integrand from `f(x, y, z)`, builds a Stroud conical quadrature with `PetscDTStroudConicalQuadrature`, reads its points and weights with `PetscQuadratureGetData`, and transports the rule from PETSc's biunit reference tetrahedron to the physical one by the affine map (barycentric coordinates and the Jacobian determinant `|det E|/8`). The number of Gauss-Jacobi points per direction is taken from the Lua global `qpts`, so the domain, the function and the quadrature strength can all be changed without recompiling. With the default `tetrahedron.lua` it integrates `x*y*z` over the tetrahedron of volume 4 and returns the exact value `0.8`.

- **ex07** — metric-based **adaptive** mesh refinement of the Fichera-corner mesh, in contrast with the uniform refinement of ex01 (`-dm_refine`, which splits every tetrahedron into 8). It builds an isotropic Riemannian metric that asks for small elements near the re-entrant corner `(0.5, 0.5, 0.5)` — where the solution of `-div(grad u) = 1` has a gradient singularity and the discretisation error concentrates — and coarse ones away from it, then calls `DMAdaptMetric()` to regenerate a mesh honouring it. It repeats this over several passes, doubling the target complexity each time to chase the singularity to ever finer scales. The mesh is read on rank 0 and then `DMPlexDistribute`d, so the adaptation has work on every rank; the final mesh is written to `fichera_adapted.vtu` for ParaView.
  - **Requires** a PETSc built with Mmg (serial) and/or ParMmg (parallel). On a PETSc without them the program still compiles and links, but `DMAdaptMetric()` returns the error *"This method requires Mmg/ParMmg"*. MareNostrum 5, or a local PETSc configured `--download-mmg --download-parmmg`, provides them.
  - **Adaptation engine, selectable at run time without recompiling** — `-dm_adaptor mmg` uses the serial Mmg; `-dm_adaptor parmmg` uses ParMmg, which remeshes in parallel across the MPI ranks (run it with `mpirun -np N`, `N > 1`; Mmg is serial). If the option is omitted, PETSc picks the first registered adaptor for the dimension, so it is best to state it explicitly.
  - **Parameters** (all optional): `-mesh_input <file>` (input mesh, default `../petsc07/cube3.msh`); `-adapt_iters <n>` (number of refinement passes, default 3); `-target_complexity <c>` (roughly the number of vertices Mmg aims for, doubled after each pass, default 15000); `-h_min` / `-h_max` (target element size near / far from the corner, default 0.01 / 0.08); `-grade_length <L>` (distance over which the target size grows from `h_min` to `h_max`, default 0.5).

When modifying or adding an example, keep the numbering scheme (`exNN.c`) and add the corresponding target to the `TARGETS` list in the Makefile.
