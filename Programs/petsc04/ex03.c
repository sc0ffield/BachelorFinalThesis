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
#include <petscdmplex.h>
#include <petscsys.h>
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>

static char help[] = "Reads a Gmsh mesh file (v2 ASCII), distributes it in parallel, and calls into Lua (functions.lua) to compute a value per process.\n\n";

int main(int argc, char **argv)
{
    DM             dm, dmDist = NULL;  // Serial mesh (dm) and its redistributed version (dmDist)
    PetscErrorCode ierr;
    PetscMPIInt    rank, size;         // Rank of this process and total number of processes
    PetscBool      flg;                // Set by PetscOptionsGetString(), unused here
    lua_State      *L;                 // Lua interpreter state, one per MPI process
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

    // Partition the mesh with METIS and redistribute it, together with its labels, across all processes
    ierr = DMPlexDistribute(dm, 0, NULL, &dmDist);
    CHKERRQ(ierr);
    if (dmDist)
    {
        ierr = DMDestroy(&dm);
        CHKERRQ(ierr);
        dm = dmDist;
    }

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

    // Start a Lua interpreter in this process and load the shared function definitions
    if ((L = luaL_newstate()) == NULL)
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_MEM, "Error creating the Lua state in this process.");

    luaL_openlibs(L);

    if (luaL_dofile(L, "functions.lua") != LUA_OK)
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Error loading the Lua file.");

    // Call compute_boundary(rank, base) in Lua, passing this process's rank and a per-rank base factor
    lua_getglobal(L, "compute_boundary");
    if (!lua_isfunction(L, -1))
    {
        PetscPrintf(PETSC_COMM_SELF, "[Process %d] Error: 'compute_boundary' not found in Lua.\n", rank);
        lua_close(L);
        return PETSC_ERR_USER;
    }
    PetscScalar baseFactor = 10.5 + (rank * 2.0);
    lua_pushinteger(L, (lua_Integer)rank);
    lua_pushnumber(L, (double)baseFactor);
    if (lua_pcall(L, 2, 1, 0) != LUA_OK)
    {
        PetscPrintf(PETSC_COMM_SELF, "[Process %d] Runtime error in compute_boundary: %s\n", rank, lua_tostring(L, -1));
        lua_close(L);
        return PETSC_ERR_USER;
    }

    double boundaryResult = 0.0;
    if (lua_isnumber(L, -1))
        boundaryResult = lua_tonumber(L, -1);
    lua_pop(L, 1);

    // Call compute_interior(rank, base) in Lua with the same arguments
    lua_getglobal(L, "compute_interior");
    if (!lua_isfunction(L, -1))
    {
        PetscPrintf(PETSC_COMM_SELF, "[Process %d] Error: 'compute_interior' not found in Lua.\n", rank);
        lua_close(L);
        return PETSC_ERR_USER;
    }

    lua_pushinteger(L, (lua_Integer)rank);
    lua_pushnumber(L, (double)baseFactor);
    if (lua_pcall(L, 2, 1, 0) != LUA_OK)
    {
        PetscPrintf(PETSC_COMM_SELF, "[Process %d] Runtime error in compute_interior: %s\n", rank, lua_tostring(L, -1));
        lua_close(L);
        return PETSC_ERR_USER;
    }
    double interiorResult = 0.0;
    if (lua_isnumber(L, -1))
        interiorResult = lua_tonumber(L, -1);

    // Each process reports its Lua-computed values and its local boundary-face count
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,
                                   "[Process %d of %d] Boundary: %.4f | Interior: %.4f (Local triangles: %d)\n",
                                   rank, size, boundaryResult, interiorResult, localBoundaryTriangles);
    if (ierr)
        return ierr;

    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);
    if (ierr)
        return ierr;

    lua_pop(L, 1);
    lua_close(L);

    ierr = DMDestroy(&dm);
    ierr = PetscFinalize();
    return ierr;
}
