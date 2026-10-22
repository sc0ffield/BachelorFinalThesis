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
#include <petscvec.h>
static char help[] = "Distributed parallel reading of a PETSc binary vector natively using MPI-IO.\n\n";

int main(int argc, char **argv)
{
    PetscErrorCode ierr;
    Vec            x;                  // Distributed vector loaded from file
    PetscViewer    viewer;             // Binary viewer used to read the file
    PetscInt       Istart, Iend;       // Range of indices owned by this process

    // Initialize PETSc and MPI
    ierr = PetscInitialize(&argc, &argv, (char*)0, help);
    if (ierr)
        return ierr;

    // Open the binary file "F.dat" for reading
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, "F.dat", FILE_MODE_READ, &viewer);
    CHKERRQ(ierr);

    // Create the vector object
    ierr = VecCreate(PETSC_COMM_WORLD, &x);
    CHKERRQ(ierr);

    // Load the vector data from the binary file (PETSc decides the parallel layout)
    ierr = VecLoad(x, viewer);
    CHKERRQ(ierr);

    // The viewer is no longer needed once the data has been loaded
    ierr = PetscViewerDestroy(&viewer);
    CHKERRQ(ierr);

    // Get the range of global indices owned by the local process
    ierr = VecGetOwnershipRange(x, &Istart, &Iend);
    CHKERRQ(ierr);

    // Each process reports the range of global indices it owns
    PetscMPIInt rank;
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    CHKERRQ(ierr);
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[Process %d] I own the range of global indices [%D, %D)\n", rank, Istart, Iend);
    CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);
    CHKERRQ(ierr);

    // Print the full contents of the distributed vector in MATLAB-like format
    ierr = PetscPrintf(PETSC_COMM_WORLD, "\nContents of the distributed vector\n");
    CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_MATLAB);
    CHKERRQ(ierr);
    ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD);
    CHKERRQ(ierr);

    // Free the vector and finalize PETSc
    ierr = VecDestroy(&x);
    CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
}
