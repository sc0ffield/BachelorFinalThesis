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
#include <petscdmplex.h>
#include <petscsys.h>
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>

static char help[] = "Reads a Gmsh mesh file (v2 ASCII), distributes it in parallel, and evaluates a configurable set of Lua functions (default: function.lua's a, b1, b2, b3, c, f) at a point.\n"
                      "Options:\n"
                      "  -mesh_input <file>   Gmsh mesh file to load (default torus.msh)\n"
                      "  -lua_input <file>    Lua file providing the functions to evaluate (default function.lua)\n\n";

int main(int argc, char **argv)
{
    DM             dm, dmDist = NULL;  // Serial mesh (dm) and its redistributed version (dmDist)
    PetscErrorCode ierr;
    PetscMPIInt    rank, size;         // Rank of this process and total number of processes
    PetscBool      flg;                // Set by PetscOptionsGetString(), unused here
    lua_State      *L;                 // Lua interpreter state, one per MPI process
    char           filename[PETSC_MAX_PATH_LEN] = "torus.msh";          // Gmsh mesh file to load
    char           luafilename[PETSC_MAX_PATH_LEN] = "function.lua";    // Lua file providing the functions to evaluate

    ierr = PetscInitialize(&argc, &argv, (char*)0, help);
    if (ierr)
        return ierr;
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    CHKERRQ(ierr);
    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);
    CHKERRQ(ierr);

    // Allow the mesh file to load to be set from the command line with -mesh_input
    ierr = PetscOptionsGetString(NULL, NULL, "-mesh_input", filename, sizeof(filename), &flg);
    CHKERRQ(ierr);

    // Allow the Lua file to load to be set from the command line with -lua_input
    // (pass e.g. -lua_input functions.lua to evaluate a different set of global functions instead)
    ierr = PetscOptionsGetString(NULL, NULL, "-lua_input", luafilename, sizeof(filename), &flg);
    CHKERRQ(ierr);

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

    // Names of the Lua functions to evaluate, each called as f(x, y, z) at the same fixed point
    const char *functionNames[] = {"a", "b1", "b2", "b3", "c", "f"};
    int numFunctions = 6;
    double results[6];

    // Start a Lua interpreter in this process and load the requested function definitions
    if ((L = luaL_newstate()) == NULL)
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_MEM, "Error creating the Lua state in this process.");

    luaL_openlibs(L);

    if (luaL_dofile(L, luafilename) != LUA_OK)
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Error loading the Lua file.");

    // Evaluate each function that actually exists in the loaded Lua file; missing ones default to 0.0
    for (int i = 0; i < numFunctions; i++)
    {
        lua_getglobal(L, functionNames[i]);
        if (!lua_isfunction(L, -1))
        {
            lua_pop(L, 1);
            continue;
        }
        lua_pushnumber(L, 1.0);
        lua_pushnumber(L, 2.5);
        lua_pushnumber(L, -2.0);
        if (lua_pcall(L, 3, 1, 0) == LUA_OK)
            results[i] = lua_tonumber(L, -1);
        else
            results[i] = 0.0;
        lua_pop(L, 1);
    }

    // Each process reports the values it computed and its local boundary-face count
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,
                                   "[Process %d of %d] a: %.4f b1: %.4f b2: %.4f b3: %.4f c: %.4f f: %.4f  (Local triangles: %d)\n",
                                   rank, size, results[0],  results[1], results[2], results[3], results[4], results[5], localBoundaryTriangles);
    if (ierr)
        return ierr;

    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);
    if (ierr)
        return ierr;

    lua_close(L);
    ierr = DMDestroy(&dm);
    ierr = PetscFinalize();
    return ierr;
}
