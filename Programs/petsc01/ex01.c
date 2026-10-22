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
static char help[] = "First example of creating and manipulating a vector in parallel with PETSc.\n\n";

int main(int argc, char **argv)
{
    PetscErrorCode ierr;
    Vec            x;                // Distributed vector
    PetscInt       n = 320;          // Global size of the vector
    PetscInt       Istart, Iend, i;  // Range of indices owned by this process
    PetscScalar    v;                // Value to insert at each index

    // Initialize PETSc and MPI
    ierr = PetscInitialize(&argc, &argv, (char*)0, help);
    if (ierr)
        return ierr;

    // Create the vector object
    ierr = VecCreate(PETSC_COMM_WORLD, &x);
    CHKERRQ(ierr);

    // Set the global size, letting PETSc decide the local size on each process
    ierr = VecSetSizes(x, PETSC_DECIDE, n);
    CHKERRQ(ierr);

    // Allow the vector type to be set from command-line options
    ierr = VecSetFromOptions(x);
    CHKERRQ(ierr);

    // Get the range of global indices owned by the local process
    ierr = VecGetOwnershipRange(x, &Istart, &Iend);
    CHKERRQ(ierr);

    // Fill the local portion of the vector with values v(i) = 10*i
    for (i = Istart; i < Iend; i++)
    {
        v = (PetscScalar)(i * 10.0);
        ierr = VecSetValues(x, 1, &i, &v, INSERT_VALUES);
        CHKERRQ(ierr);
    }

    // Assemble the vector so it can be used (communicates values across processes)
    ierr = VecAssemblyBegin(x);
    CHKERRQ(ierr);
    ierr = VecAssemblyEnd(x);
    CHKERRQ(ierr);

    // Print the vector contents to standard output
    ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD);
    CHKERRQ(ierr);

    // Free the vector and finalize PETSc
    ierr = VecDestroy(&x);
    CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
}
