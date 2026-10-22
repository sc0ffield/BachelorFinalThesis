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
#include <petscdmplex.h>
#include <petscsys.h>

static char help[] = "Reads a Gmsh mesh file (v2 ASCII) and distributes it in parallel.\n\n";

int main(int argc, char **argv)
{
    DM             dm, dmDist = NULL;  // Serial mesh (dm) and its redistributed version (dmDist)
    PetscErrorCode ierr;
    PetscMPIInt    rank, size;         // Rank of this process and total number of processes
    PetscInt       numCells;           // Number of local cells owned by this process after distribution
    PetscBool      flg;                // Set by PetscOptionsGetString(), unused here
    char           filename[PETSC_MAX_PATH_LEN] = "torus.msh"; // Gmsh mesh file to load

    ierr = PetscInitialize(&argc, &argv, (char*)0, help);
    if (ierr)
        return ierr;

    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    CHKERRQ(ierr);
    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);
    CHKERRQ(ierr);

    if (rank == 0)
    {
        PetscPrintf(PETSC_COMM_SELF, "========================================================\n");
        PetscPrintf(PETSC_COMM_SELF, " Running in parallel with %d processes\n", size);
        PetscPrintf(PETSC_COMM_SELF, "========================================================\n");
    }

    // Allow the mesh file to load to be set from the command line with -mesh_input
    ierr = PetscOptionsGetString(NULL, NULL, "-mesh_input", filename, sizeof(filename), &flg);
    CHKERRQ(ierr);

    if (rank == 0)
        PetscPrintf(PETSC_COMM_SELF, "[Process 0] Reading mesh file: %s...\n", filename);

    // Read the Gmsh mesh on process 0 and build a serial DMPlex from it.
    // interpolate = PETSC_TRUE also creates all the faces and edges, which the
    // Gmsh boundary labels ("Face Sets") attach to and which regular refinement
    // needs: -dm_refine below requires a fully interpolated mesh.
    ierr = DMPlexCreateGmshFromFile(PETSC_COMM_WORLD, filename, PETSC_TRUE, &dm);
    CHKERRQ(ierr);

    // Let runtime options act on the serial mesh before it is distributed.
    // The relevant one here is -dm_refine <n>: it uniformly refines the mesh,
    // splitting every tetrahedron into 8, so n levels multiply the cell count
    // by 8^n. Refining before DMPlexDistribute keeps the refinement serial and
    // cheap, and the finer mesh is then partitioned across the processes.
    ierr = DMSetFromOptions(dm);
    CHKERRQ(ierr);

    if (rank == 0)
        PetscPrintf(PETSC_COMM_SELF, "[Process 0] Partitioning and distributing the mesh with METIS...\n");

    // Partition the mesh with METIS and redistribute it across all processes
    ierr = DMPlexDistribute(dm, 0, NULL, &dmDist);
    CHKERRQ(ierr);

    if (dmDist)
    {
        ierr = DMDestroy(&dm);
        CHKERRQ(ierr);
        dm   = dmDist;
    }

    // Count the local cells (mesh elements at height 0, i.e. tetrahedra) owned by this process
    PetscInt cellStart, cellEnd;
    ierr = DMPlexGetHeightStratum(dm, 0, &cellStart, &cellEnd);
    CHKERRQ(ierr);
    numCells = cellEnd - cellStart;

    // Each process reports how many cells it now owns; output is flushed together, in rank order
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[Process %d] Holding a local block of %d tetrahedra in memory.\n", rank, numCells);
    CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);
    CHKERRQ(ierr);

    if (rank == 0)
        PetscPrintf(PETSC_COMM_SELF, "\n--- Summary of the DM's parallel structure ---\n");
    ierr = DMView(dm, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

    ierr = DMDestroy(&dm);
    CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
}
