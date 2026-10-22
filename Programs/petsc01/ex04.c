/**************************************************************************************
* Filename:   ex04.c
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

static char help[] = "Advanced example (ex04) combining VecLoad, ghosted vectors and raw access to local RAM.\n\n";

int main(int argc, char **argv)
{
    PetscErrorCode ierr;
    Vec            x_in, x_ghost;      // Loaded vector and its ghosted counterpart
    PetscViewer    viewer;             // Binary viewer used to read the file
    PetscInt       n_global, n_local, Istart, Iend;
    PetscInt       nghost = 1;         // Number of ghost (halo) entries requested
    PetscInt       ifantasma[1];       // Global indices of the ghost entries
    PetscScalar    *x_array;
    PetscMPIInt    rank, size;

    // Initialize PETSc and MPI
    ierr = PetscInitialize(&argc, &argv, (char*)0, help);
    if (ierr)
        return ierr;
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    CHKERRQ(ierr);
    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);
    CHKERRQ(ierr);

    // Load the vector from the binary file "F.dat"
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, "F.dat", FILE_MODE_READ, &viewer);
    CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_WORLD, &x_in);
    CHKERRQ(ierr);
    ierr = VecLoad(x_in, viewer);
    CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);
    CHKERRQ(ierr);

    // Find out the sizes and the range that PETSc has decided in parallel
    ierr = VecGetSize(x_in, &n_global);
    CHKERRQ(ierr);
    ierr = VecGetLocalSize(x_in, &n_local);
    CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(x_in, &Istart, &Iend);
    CHKERRQ(ierr);

    // Each process (except the last) needs one ghost entry: its right neighbor
    if (Iend < n_global)
        ifantasma[0] = Iend;
    else
        nghost = 0; // The last process does not need a right-hand neighbor

    // Create a ghosted vector with the same local layout as x_in plus the ghost entry
    ierr = VecCreateGhost(PETSC_COMM_WORLD, n_local, PETSC_DECIDE, nghost, ifantasma, &x_ghost);
    CHKERRQ(ierr);

    // Copy the loaded values into the ghosted vector
    ierr = VecCopy(x_in, x_ghost);
    CHKERRQ(ierr);

    // Exchange ghost values between neighboring processes
    ierr = VecGhostUpdateBegin(x_ghost, INSERT_VALUES, SCATTER_FORWARD);
    CHKERRQ(ierr);
    ierr = VecGhostUpdateEnd(x_ghost, INSERT_VALUES, SCATTER_FORWARD);
    CHKERRQ(ierr);

    // Get direct access to the local array, including the ghost entry
    ierr = VecGetArray(x_ghost, &x_array);
    CHKERRQ(ierr);

    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Process %d [Global range %D to %D)\n", rank, Istart, Iend);
    CHKERRQ(ierr);

    // For each local element, print it together with the value of its right-hand neighbor
    for (PetscInt i = 0; i < n_local; i++)
    {
        PetscInt index_global = Istart + i;
        PetscScalar valor_local = x_array[i];

        if (index_global < n_global - 1)
        {
            PetscScalar valor_vei_dreta = x_array[n_local + 0]; // The first ghost entry is at position n_local
            ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "  Global element %D (%g) -> Its right neighbor is (%g)\n",
                                           index_global, (double)valor_local, (double)valor_vei_dreta);
            CHKERRQ(ierr);
        }
        else
        {
            ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "  Global element %D (%g) -> I am the very end of the array.\n",
                                           index_global, (double)valor_local); CHKERRQ(ierr);
        }
    }
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);
    CHKERRQ(ierr);

    // Release direct array access
    ierr = VecRestoreArray(x_ghost, &x_array);
    CHKERRQ(ierr);

    // Free the vectors and finalize PETSc
    ierr = VecDestroy(&x_in);
    CHKERRQ(ierr);
    ierr = VecDestroy(&x_ghost);
    CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
}