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
#include <petscmat.h>
static char help[] = "First example (ex01): creation and assembly of a distributed sparse matrix in PETSc.\n"
                      "Options:\n"
                      "  -n <size>   Global size of the square matrix (default 10)\n\n";

int main(int argc, char **argv)
{
    PetscErrorCode ierr;
    Mat            A;                  // Distributed sparse matrix
    PetscInt       n = 10;             // Global size of the matrix (n x n)
    PetscInt       Istart, Iend, i;    // Range of rows owned by this process

    // Initialize PETSc and MPI
    ierr = PetscInitialize(&argc, &argv, (char*)0, help);
    if (ierr)
        return ierr;

    // Allow the matrix size to be set from the command line with -n
    ierr = PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL);
    CHKERRQ(ierr);

    // Create the matrix object
    ierr = MatCreate(PETSC_COMM_WORLD, &A);
    CHKERRQ(ierr);

    // Set the global size, letting PETSc decide the local row distribution
    ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n);
    CHKERRQ(ierr);

    // Set the AIJ type (sparse, works both sequentially and in parallel)
    ierr = MatSetType(A, MATAIJ);
    CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);
    CHKERRQ(ierr);

    /*
        Number of processes equal to 1
        mpirun -np 1
        This is a tridiagonal matrix: 3 nonzeros per row must be preallocated
    */
    ierr = MatSeqAIJSetPreallocation(A, 3, NULL);
    CHKERRQ(ierr);

    /*
        Number of processes greater than 1
        mpirun -np 12
        This is a tridiagonal matrix: 3 nonzeros in the diagonal block
        and 3 nonzeros in the off-diagonal block per row
    */
    ierr = MatMPIAIJSetPreallocation(A, 3, NULL, 3, NULL);
    CHKERRQ(ierr);

    // Get the range of global rows owned by this process
    ierr = MatGetOwnershipRange(A, &Istart, &Iend);
    CHKERRQ(ierr);

    // Fill the local rows: main diagonal with 2.0 and the two neighboring diagonals with -1.0
    for (i = Istart; i < Iend; i++)
    {
        PetscScalar val_diagonal = 2.0;
        PetscScalar val_neighbor = -1.0;
        PetscInt    col;

        if (i > 0)
        {
            col = i - 1;
            ierr = MatSetValues(A, 1, &i, 1, &col, &val_neighbor, INSERT_VALUES);
            CHKERRQ(ierr);
        }

        col = i;
        ierr = MatSetValues(A, 1, &i, 1, &col, &val_diagonal, INSERT_VALUES);
        CHKERRQ(ierr);

        if (i < n - 1)
        {
            col = i + 1;
            ierr = MatSetValues(A, 1, &i, 1, &col, &val_neighbor, INSERT_VALUES);
            CHKERRQ(ierr);
        }
    }

    // Assemble the matrix (communicates values across processes)
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);

    // Print the matrix contents to standard output
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Distributed Tridiagonal AIJ Matrix (%Dx%D)\n", n, n);
    CHKERRQ(ierr);
    ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);
    CHKERRQ(ierr);

    // Free the matrix and finalize PETSc
    ierr = MatDestroy(&A);
    CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
}
