/**************************************************************************************
* Filename:   ex01.c
* Author:     Marina-Liao Colom (marina-liao.colom AT estudiantat.upc.edu) and Rafel Amer (rafel.amer AT upc.edu)
* Copyright:  Marina-Liao Colom and Rafel Amer 2026
* Assisted-by: Developed with the assistance of Claude Code (Anthropic), https://claude.com/claude-code
* Disclaimer: This code is presented "as is" and it has been written for
* educational purposes.
*
* License:    This program is free software; you can redistribute it and/or
* modify it under the terms of either:
*
* 1 the GNU Lesser General Public License as published by the Free
* Software Foundation; either version 3 of the License, or (at your
* option) any later version.
*
* or
*
* 2 the GNU General Public License as published by the Free Software
* Foundation; either version 2 of the License, or (at your option)
* any later version.
*
*	      See https://www.gnu.org/licenses/
***************************************************************************************/
#include <petscvec.h>
#include <petscmat.h>
#include <petscksp.h>

static char help[] = "Parallel distributed reading of a PETSc binary vector and matrix using native MPI-IO, and solution of the resulting linear system.\n\n";

int main(int argc, char **argv)
{
    PetscErrorCode ierr;
    Vec            F, X;               // Right-hand side vector F and solution vector X
    Mat            K;                  // Distributed sparse matrix, loaded from disk
    PetscViewer    viewer;             // Binary viewer used to read F.dat and K.dat
    KSP            ksp;                // Krylov linear solver context

    ierr = PetscInitialize(&argc, &argv, (char*)0, help);
    if (ierr)
        return ierr;

    // Open F.dat and load the right-hand side vector in parallel via MPI-IO
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, "F.dat", FILE_MODE_READ, &viewer);
    CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_WORLD, &F);
    CHKERRQ(ierr);
    ierr = VecLoad(F, viewer);
    CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);
    CHKERRQ(ierr);

    // Open K.dat and load the sparse matrix in parallel via MPI-IO
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, "K.dat", FILE_MODE_READ, &viewer);
    CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD, &K);
    CHKERRQ(ierr);
    ierr = MatSetType(K, MATAIJ);
    CHKERRQ(ierr);
    ierr = MatLoad(K, viewer);
    CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);
    CHKERRQ(ierr);

    /*
        Let's solve KX = F
    */
    // Allocate X with the same layout as F and set up the KSP solver for K
    ierr = VecDuplicate(F, &X);
    CHKERRQ(ierr);
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);
    CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp, K, K);
    CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);
    CHKERRQ(ierr);
    ierr = KSPSolve(ksp, F, X);
    CHKERRQ(ierr);

    // Report how many iterations the solver needed to converge
    PetscInt its;
    ierr = KSPGetIterationNumber(ksp, &its);
    CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD, "Total number of iterations: %" PetscInt_FMT "\n", its);

    // Free all PETSc objects and finalize
    ierr = KSPDestroy(&ksp);
    CHKERRQ(ierr);
    ierr = VecDestroy(&X);
    CHKERRQ(ierr);
    ierr = MatDestroy(&K);
    CHKERRQ(ierr);
    ierr = VecDestroy(&F);
    CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
}
