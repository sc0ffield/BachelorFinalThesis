/**************************************************************************************
* Filename:   ex05.c
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

static char help[] = "Advanced example (ex05) with 3 ghost elements on the left and 3 on the right, using VecGhostGetLocalForm.\n\n";

int main(int argc, char **argv)
{
    PetscErrorCode ierr;
    Vec            x_in, x_ghost, x_local_form; // Loaded vector, ghosted vector and its local form
    PetscViewer    viewer;                      // Binary viewer used to read the file
    PetscInt       n_global, n_local, Istart, Iend;
    PetscInt       nghost = 0;                  // Actual number of ghost entries used
    PetscInt       ifantasma[6];                // Global indices of the ghost entries (up to 3 left + 3 right)
    PetscScalar    *x_array;
    PetscMPIInt    rank;
    PetscInt       i, j;

    // Initialize PETSc and MPI
    ierr = PetscInitialize(&argc, &argv, (char*)0, help);
    if (ierr)
        return ierr;
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
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

    // Collect up to 3 ghost indices to the left of the local range (skipping those out of bounds)
    for (j = 3; j >= 1; j--)
    {
        PetscInt idx = Istart - j;
        if (idx >= 0)
        {
            ifantasma[nghost] = idx;
            nghost++;
        }
    }
    // Collect up to 3 ghost indices to the right of the local range (skipping those out of bounds)
    for (j = 0; j < 3; j++)
    {
        PetscInt idx = Iend + j;
        if (idx < n_global)
        {
            ifantasma[nghost] = idx;
            nghost++;
        }
    }

    // Create a ghosted vector with the same local layout as x_in plus the collected ghost entries
    ierr = VecCreateGhost(PETSC_COMM_WORLD, n_local, PETSC_DECIDE, nghost, ifantasma, &x_ghost);
    CHKERRQ(ierr);
    ierr = VecCopy(x_in, x_ghost);
    CHKERRQ(ierr);

    // Exchange ghost values between neighboring processes
    ierr = VecGhostUpdateBegin(x_ghost, INSERT_VALUES, SCATTER_FORWARD);
    CHKERRQ(ierr);
    ierr = VecGhostUpdateEnd(x_ghost, INSERT_VALUES, SCATTER_FORWARD);
    CHKERRQ(ierr);

    // Get the local form of the ghosted vector (local entries followed by ghost entries)
    ierr = VecGhostGetLocalForm(x_ghost, &x_local_form);
    CHKERRQ(ierr);

    // Get direct access to the local array, including the ghost entries
    ierr = VecGetArray(x_local_form, &x_array);
    CHKERRQ(ierr);

    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "=== Process %d [Owns indices %D to %D) ===\n", rank, Istart, Iend);
    CHKERRQ(ierr);

    PetscInt i_local = 0;
    PetscInt i_global = Istart + i_local;

    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "  Central element studied: Global index %D (Value: %.4f)\n", i_global, (double)x_array[i_local]);
    CHKERRQ(ierr);

    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "    Left ghosts in RAM:\n");
    CHKERRQ(ierr);
    PetscInt desplaçament = n_local;

    if (Istart >= 3)
    {
        ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "      3 positions back (-3): %.4f\n", (double)x_array[desplaçament + 0]);
        CHKERRQ(ierr);
        ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "      2 positions back (-2): %.4f\n", (double)x_array[desplaçament + 1]);
        CHKERRQ(ierr);
        ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "      1 position back  (-1): %.4f\n", (double)x_array[desplaçament + 2]);
        CHKERRQ(ierr);
        desplaçament += 3; // Advance the ghost-reading pointer
    }
    else
    {
        ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "      (There are no 3 elements to the left of the absolute start)\n");
        if (Istart == 2) desplaçament += 2;
        if (Istart == 1) desplaçament += 1;
    }

    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "    Right ghosts in RAM:\n"); CHKERRQ(ierr);
    if (Iend <= n_global - 3)
    {
        ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "      1 position forward (+1): %.4f\n", (double)x_array[desplaçament + 0]);
        CHKERRQ(ierr);
        ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "      2 positions forward (+2): %.4f\n", (double)x_array[desplaçament + 1]);
        CHKERRQ(ierr);
        ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "      3 positions forward (+3): %.4f\n", (double)x_array[desplaçament + 2]);
        CHKERRQ(ierr);
    } else {
        ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "      (There are no 3 elements to the right of the absolute end)\n");
    }

    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT); 
    CHKERRQ(ierr);

    // Release direct array access and the local form
    ierr = VecRestoreArray(x_local_form, &x_array);
    CHKERRQ(ierr);
    ierr = VecGhostRestoreLocalForm(x_ghost, &x_local_form);
    CHKERRQ(ierr);

    // Free the vectors and finalize PETSc
    ierr = VecDestroy(&x_in);
    CHKERRQ(ierr);
    ierr = VecDestroy(&x_ghost);
    CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
}
