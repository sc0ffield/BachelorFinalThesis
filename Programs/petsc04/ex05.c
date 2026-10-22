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
#include <petscdmplex.h>
#include <petscsys.h>
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>

static char help[] = "Reads a Gmsh mesh file (v2 ASCII), distributes it in parallel, and evaluates Lua functions (default: function.lua's a, b1, b2, b3, c, f) at the centroid of every local cell.\n"
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
    ierr = PetscOptionsGetString(NULL, NULL, "-lua_input", luafilename, sizeof(filename), &flg);
    CHKERRQ(ierr);

    // Read the Gmsh mesh; it must be interpolated (faces/edges built) for DMPlexComputeCellGeometryFVM to work
    ierr = DMPlexCreateGmshFromFile(PETSC_COMM_WORLD, filename, PETSC_TRUE, &dm);
    CHKERRQ(ierr);

    // Partition the mesh with METIS and redistribute it across all processes
    ierr = DMPlexDistribute(dm, 0, NULL, &dmDist);
    CHKERRQ(ierr);
    if (dmDist)
    {
        ierr = DMDestroy(&dm);
        CHKERRQ(ierr);
        dm = dmDist;
    }

    // Range of local cells (height 0: tetrahedra) owned by this process
    PetscInt cellStart, cellEnd;
    ierr = DMPlexGetHeightStratum(dm, 0, &cellStart, &cellEnd);
    CHKERRQ(ierr);
    PetscInt numLocalCells = cellEnd - cellStart;

    // Names of the Lua functions to evaluate, each called as f(x, y, z) at a cell centroid
    const char *functionNames[] = {"a", "b1", "b2", "b3", "c", "f"};
    int    numFunctions = 6;
    double sums[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // Start a Lua interpreter in this process and load the requested function definitions
    if ((L = luaL_newstate()) == NULL)
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_MEM, "Error creating the Lua state in this process.");

    luaL_openlibs(L);

    if (luaL_dofile(L, luafilename) != LUA_OK)
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Error loading the Lua file.");

    // For every local cell, compute its centroid and evaluate each Lua function there,
    // accumulating a running sum per function to report an average over all local cells
    for (PetscInt c = cellStart; c < cellEnd; ++c)
    {
        PetscReal volume, centroid[3], normal[3];
        ierr = DMPlexComputeCellGeometryFVM(dm, c, &volume, centroid, normal);
        CHKERRQ(ierr);

        for (int i = 0; i < numFunctions; i++)
        {
            lua_getglobal(L, functionNames[i]);
            if (!lua_isfunction(L, -1))
            {
                lua_pop(L, 1);
                continue;
            }
            lua_pushnumber(L, (double)centroid[0]);
            lua_pushnumber(L, (double)centroid[1]);
            lua_pushnumber(L, (double)centroid[2]);
            if (lua_pcall(L, 3, 1, 0) == LUA_OK)
                sums[i] += lua_tonumber(L, -1);
            lua_pop(L, 1);
        }
    }

    // Each process reports the average of each function over the centroids of its local cells
    double average[6];
    for (int i = 0; i < numFunctions; i++)
        average[i] = (numLocalCells > 0) ? sums[i] / numLocalCells : 0.0;

    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,
                                   "[Process %d of %d] Averages over %d local cells -> a: %.4f b1: %.4f b2: %.4f b3: %.4f c: %.4f f: %.4f\n",
                                   rank, size, numLocalCells, average[0], average[1], average[2], average[3], average[4], average[5]);
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
