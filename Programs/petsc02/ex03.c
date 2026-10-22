/**************************************************************************************
* Filename:   ex03.c
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

static char help[] = "Advanced example (ex03): parallel reading of the binary matrix file K.dat with MPI-IO.\n"
                      "Options:\n"
                      "  -f <file>   PETSc binary file to load (default K.dat)\n\n";

int main(int argc, char **argv)
{
    PetscErrorCode ierr;
    Mat            K;                  // Distributed sparse matrix, loaded from disk
    PetscViewer    viewer;             // Binary viewer associated with the matrix file
    PetscInt       m, n;               // Global size of the matrix (m rows x n columns)
    PetscInt       Istart, Iend;       // Range of rows owned by this process
    PetscMPIInt    rank, size;         // Rank of this process and total number of processes
    char           filename[PETSC_MAX_PATH_LEN] = "K.dat"; // Name of the binary file to load

    // Initialize PETSc and MPI
    ierr = PetscInitialize(&argc, &argv, (char*)0, help);
    if (ierr)
        return ierr;
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    CHKERRQ(ierr);
    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);
    CHKERRQ(ierr);

    // Allow the file to load to be set from the command line with -f
    ierr = PetscOptionsGetString(NULL, NULL, "-f", filename, sizeof(filename), NULL);
    CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD, "Starting to read '%s' with %d processes...\n", filename, size);
    CHKERRQ(ierr);

    // Open the binary file for reading through a PetscViewer
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ, &viewer);
    CHKERRQ(ierr);

    // Create the matrix and load it in parallel from the viewer via MPI-IO
    ierr = MatCreate(PETSC_COMM_WORLD, &K);
    CHKERRQ(ierr);
    ierr = MatSetType(K, MATAIJ);
    CHKERRQ(ierr);
    ierr = MatLoad(K, viewer);
    CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);
    CHKERRQ(ierr);

    // Get the global size and the range of rows owned by this process
    ierr = MatGetSize(K, &m, &n);
    CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(K, &Istart, &Iend);
    CHKERRQ(ierr);

    // Each process writes its line to the synchronized buffer; they are flushed together, in order
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[Process %d] Handles global rows %D through %D (%D local rows in total).\n",
                                   rank, Istart, Iend, Iend - Istart);
    CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);
    CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD, "\nREAD COMPLETED SUCCESSFULLY\n");
    CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Global matrix size: %D rows x %D columns\n", m, n);
    CHKERRQ(ierr);

    // Print the contents of the loaded matrix
    ierr = MatView(K, PETSC_VIEWER_STDOUT_WORLD);
    CHKERRQ(ierr);

    // Free the matrix and finalize PETSc
    ierr = MatDestroy(&K);
    CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
}
