/**************************************************************************************
* Filename:   ex02.c
* Author:     Marina-Liao Colom (marina-liao.colom AT estudiantat.upc.edu) and Rafel Amer (rafel.amer AT upc.edu)
* Copyright:  Marina-Liao Colom and Rafel Amer 2026
* Assisted-by: Developed with the assistance of Claude Code (Anthropic), https://claude.com/claude-code
* Disclaimer: This code is presented "as is" and it has been written for
* educational purposes.
*
* License: This program is free software; you can redistribute it and/or
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
    PetscBool      flg;                // Set by PetscOptionsGetString(), unused here
    char           filename[PETSC_MAX_PATH_LEN] = "torus.msh"; // Gmsh mesh file to load

    ierr = PetscInitialize(&argc, &argv, (char*)0, help);
    if (ierr)
        return ierr;
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    CHKERRQ(ierr);
    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);
    CHKERRQ(ierr);

    // Allow the mesh file to load to be set from the command line with -mesh_input
    ierr = PetscOptionsGetString(NULL, NULL, "-mesh_input", filename, sizeof(filename), &flg); CHKERRQ(ierr);

    // Read the Gmsh mesh (including physical/boundary tags, PETSC_TRUE) and let command-line options apply
    ierr = DMPlexCreateGmshFromFile(PETSC_COMM_WORLD, filename, PETSC_TRUE, &dm);
    CHKERRQ(ierr);

    ierr = DMSetFromOptions(dm);
    CHKERRQ(ierr);

    if (rank == 0)
        PetscPrintf(PETSC_COMM_SELF, "[Process 0] Distributing the mesh and its labels...\n");

    // Partition the mesh with METIS and redistribute it, together with its labels, across all processes
    ierr = DMPlexDistribute(dm, 0, NULL, &dmDist);
    CHKERRQ(ierr);
    if (dmDist)
    {
        ierr = DMDestroy(&dm);
        CHKERRQ(ierr);
        dm = dmDist;
    }

    if (rank == 0)
        PetscPrintf(PETSC_COMM_SELF, "\n--- INSPECTING BOUNDARY LABEL VALUES ---\n");

    // Gmsh physical groups are usually imported as the "Face Sets" label; fall back to "marker" otherwise
    DMLabel boundaryLabel;
    ierr = DMGetLabel(dm, "Face Sets", &boundaryLabel);
    CHKERRQ(ierr);

    if (!boundaryLabel)
    {
        ierr = DMGetLabel(dm, "marker", &boundaryLabel);
        CHKERRQ(ierr);
        if (boundaryLabel && rank == 0)
            PetscPrintf(PETSC_COMM_SELF, "[WARNING] 'Face Sets' not found, using the native 'marker' label instead.\n");
    }

    if (!boundaryLabel)
    {
        if (rank == 0)
        {
            PetscPrintf(PETSC_COMM_SELF, "[ERROR] Neither 'Face Sets' nor 'marker' was found. Listing the actual labels present in the DM:\n");
            ierr = DMView(dm, PETSC_VIEWER_STDOUT_SELF);
            CHKERRQ(ierr);
        }

        ierr = DMDestroy(&dm);
        ierr = PetscFinalize();
        return ierr;
    }

    // Walk the faces (height 1) and count how many carry a boundary label value (!= -1)
    PetscInt faceStart, faceEnd;
    PetscInt localBoundaryTriangles = 0;

    ierr = DMPlexGetHeightStratum(dm, 1, &faceStart, &faceEnd);
    CHKERRQ(ierr);

    for (PetscInt f = faceStart; f < faceEnd; ++f)
    {
        PetscInt labelValue;
        ierr = DMLabelGetValue(boundaryLabel, f, &labelValue);
        CHKERRQ(ierr);

        if (labelValue != -1)
            localBoundaryTriangles++;
    }

    // Each process reports its local boundary-face count; output is flushed together, in rank order
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,
        "[Process %d] Handling %d boundary triangles (Face Sets).\n",
        rank, localBoundaryTriangles); CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);
    CHKERRQ(ierr);

    if (rank == 0)
    {
        PetscPrintf(PETSC_COMM_SELF, "\n[Process 0] Summary of subregions detected by PETSc:\n");
        ierr = DMLabelView(boundaryLabel, PETSC_VIEWER_STDOUT_SELF);
        CHKERRQ(ierr);
    }

    ierr = DMDestroy(&dm);
    ierr = PetscFinalize();
    return ierr;
}
