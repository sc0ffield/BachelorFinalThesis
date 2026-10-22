/**************************************************************************************
* Filename:   ex02.c
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
static char help[] = "Parallel algebraic operations with vectors in PETSc.\n\n";

int main(int argc, char **argv)
{
    PetscErrorCode ierr;
    Vec            x, y;               // Two distributed vectors
    PetscInt       n = 10000;          // Global size of the vectors
    PetscInt       Istart, Iend, i;    // Range of indices owned by this process
    PetscScalar    alpha = 0.5, dot_product;
    PetscReal      norm_y;

    // Initialize PETSc and MPI
    ierr = PetscInitialize(&argc, &argv, (char*)0, help);
    if (ierr)
        return ierr;

    // Create vector x and let its type be set from command-line options
    ierr = VecCreate(PETSC_COMM_WORLD, &x);
    CHKERRQ(ierr);
    ierr = VecSetSizes(x, PETSC_DECIDE, n);
    CHKERRQ(ierr);
    ierr = VecSetFromOptions(x);
    CHKERRQ(ierr);

    // Create vector y with the same layout and type as x
    ierr = VecDuplicate(x, &y);
    CHKERRQ(ierr);

    // Get the range of global indices owned by the local process
    ierr = VecGetOwnershipRange(x, &Istart, &Iend);
    CHKERRQ(ierr);

    // Fill the local portion of x with values x(i) = i
    for (i = Istart; i < Iend; i++)
    {
        PetscScalar value = (PetscScalar)i;
        ierr = VecSetValues(x, 1, &i, &value, INSERT_VALUES);
        CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(x);
    CHKERRQ(ierr);
    ierr = VecAssemblyEnd(x);
    CHKERRQ(ierr);

    // Set every entry of y to 1.0
    ierr = VecSet(y, 1.0);
    CHKERRQ(ierr);

    // Scale x by alpha: x = alpha * x
    ierr = VecScale(x, alpha); CHKERRQ(ierr);

    /*
        AXPY operation: y = alpha * x + y  (here we compute y = 1.0 * x + y)
    */
    ierr = VecAXPY(y, 1.0, x);
    CHKERRQ(ierr);

    // Compute the dot product (x, y)
    ierr = VecDot(x, y, &dot_product);
    CHKERRQ(ierr);

    // Compute the L2 norm of y
    ierr = VecNorm(y, NORM_2, &norm_y);
    CHKERRQ(ierr);

    // Only process 0 prints the final results
    PetscMPIInt rank;
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    CHKERRQ(ierr);
    if (rank == 0) {
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Results of the operations\n");
        CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Dot product (x, y)   = %g\n", (double)PetscRealPart(dot_product));
        CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "L2 norm of vector y  = %g\n", (double)norm_y);
        CHKERRQ(ierr);
    }

    // Free the vectors and finalize PETSc
    ierr = VecDestroy(&x);
    CHKERRQ(ierr);
    ierr = VecDestroy(&y);
    CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
}
