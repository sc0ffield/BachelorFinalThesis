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
#include <petscmat.h>

static char help[] = "Example (ex02): distributed matrix-vector multiplication (y = A * x) in PETSc.\n"
                      "Options:\n"
                      "  -n <size>   Global size of the square matrix and of the vectors (default 10)\n\n";

int main(int argc, char **argv)
{
    PetscErrorCode ierr;
    Mat            A;                  // Distributed sparse matrix
    Vec            x, y;               // Distributed vectors (x = operand, y = result)
    PetscInt       n = 10;             // Global size of the matrix and of the vectors
    PetscInt       Istart, Iend, i;    // Range of rows owned by this process

    // Initialize PETSc and MPI
    ierr = PetscInitialize(&argc, &argv, (char*)0, help);
    if (ierr)
        return ierr;

    // Allow the matrix/vector size to be set from the command line with -n
    ierr = PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL);
    CHKERRQ(ierr);

    /*
        Matrix A: the same tridiagonal matrix as in example ex01
    */
    ierr = MatCreate(PETSC_COMM_WORLD, &A);
    CHKERRQ(ierr);
    ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n);
    CHKERRQ(ierr);
    ierr = MatSetType(A, MATAIJ);
    CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);
    CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(A, 3, NULL);
    CHKERRQ(ierr);
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

    /*
        Vectors x and y
        MatCreateVecs is much better than VecCreate because it guarantees
        that the vectors have exactly the same row distribution as the matrix A.
    */
    ierr = MatCreateVecs(A, &x, &y);
    CHKERRQ(ierr);
    ierr = VecSet(x, 1.0);
    CHKERRQ(ierr);

    // Compute the matrix-vector product y = A * x
    ierr = MatMult(A, x, y);
    CHKERRQ(ierr);

    // Print the result vector in MATLAB ASCII format
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Result Vector y (y = A * x)\n");
    CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_MATLAB);
    CHKERRQ(ierr);
    ierr = VecView(y, PETSC_VIEWER_STDOUT_WORLD);
    CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);
    CHKERRQ(ierr);

    // Free the matrix and vectors, and finalize PETSc
    ierr = MatDestroy(&A);
    CHKERRQ(ierr);
    ierr = VecDestroy(&x);
    CHKERRQ(ierr);
    ierr = VecDestroy(&y);
    CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
}
