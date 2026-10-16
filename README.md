# PETSc

Working repository for the final degree thesis **"Study and Implementation of a
Three-Dimensional Numerical Simulator Based on the Finite Element Method in
Parallel Computing Environments"**, at
[ESEIAAT](https://eseiaat.upc.edu/) (Escola Superior d'Enginyeries Industrial,
Aeroespacial i Audiovisual de Terrassa), Universitat Politècnica de Catalunya
(UPC), by Marina-Liao Colom Escolá, supervised by Rafel Amer Ramon.

## Description

The thesis studies and implements a parallel numerical simulator for boundary
value problems posed on three-dimensional domains and discretized with the
finite element method (FEM). Building on [PETSc](https://petsc.org/)'s
distributed data structures (`Vec`, `Mat`, `DMPlex`) and solver frameworks
(`KSP`, `SNES`, `TS`), the simulator:

- reads unstructured tetrahedral meshes generated with [Gmsh](https://gmsh.info/);
- assembles and solves both steady (elliptic) and time-dependent (parabolic)
  PDEs with `PetscFE`;
- lets the boundary conditions, coefficients and source terms of each problem
  be specified through an embedded [Lua](https://www.lua.org/) scripting
  layer, without recompiling the solver;
- runs in parallel via MPI and domain decomposition; and
- is validated against known analytical or reference solutions and
  benchmarked in sequential vs. parallel configurations.

This repository tracks the incremental development of that simulator.

## Proposed table of contents

1. **Preface**
   1. Abstract
   2. Aim
   3. Scope
   4. Specifications
   5. Reason to be
   6. State of the art

2. **The Boundary Value Problem in 3D**
   1. The steady equation
   2. The time-dependent equation
   3. Boundary conditions
   4. Initial conditions and well-posedness of transient problems
   5. Physical modelling

3. **Mathematical Tools for 3D and Time-Dependent Problems**
   1. Polynomial interpolation on tetrahedra
   2. Quadrature on tetrahedra
   3. Time discretization schemes
   4. Introduction to PETSc

4. **Three-Dimensional Domains and Meshing**
   1. From in-house triangulation to external mesh generators
   2. The `.msh` format and `DMPlexCreateGmshFromFile`
   3. Boundary tagging
   4. Parallel mesh distribution
   5. Mesh refinement

5. **The Finite Element Method in Dimension Three**
   1. Weak form of the steady problem
   2. Weak form of the time-dependent problem
   3. The discrete system: residual and Jacobian
   4. Auxiliary fields
   5. Imposing boundary conditions with `PetscDSAddBoundary`
   6. Time integration of the semi-discrete system

6. **Problem Specification with Lua**
   1. Design of the Lua specification language
   2. Time-dependent coefficients and source terms in Lua
   3. Embedding the Lua interpreter in the C/PETSc code

7. **Parallel Solvers with PETSc**
   1. Distributed data structures, MPI and domain decomposition
   2. SNES/KSP for the steady, linear case
   3. TS for the time-dependent case
   4. Preconditioners and scalability
   5. Output

8. **Examples and Validation**
   1. Steady Laplace/Poisson problem on a sphere or cube
   2. Steady advection–diffusion–reaction problem on a torus
   3. Transient heat diffusion problem
   4. Sequential vs. parallel performance comparison
   5. Comparison with the 2D solver from the original project

9. **Conclusion**
   1. Further reading
   2. Further work

10. **Bibliography**

**Appendix. The `petsc` example programs** — a guided walkthrough of the
`Programs/petsc01`–`petsc08` folders, gathering their `README.md` files into a
single narrative; typeset as a standalone document (`LaTeX/tfe-appendix.tex`).
