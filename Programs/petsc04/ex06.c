/**************************************************************************************
* Filename:   ex06.c
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
#include <petscdt.h>
#include <petscsys.h>
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>

static char help[] = "Integrates a function over a tetrahedron, both defined in a Lua file.\n"
                     "The tetrahedron vertices (table 'vertices') and the integrand (function\n"
                     "f(x,y,z)) are read from the Lua file; the integral is computed with PETSc's\n"
                     "Stroud conical simplex quadrature, transported to the physical tetrahedron\n"
                     "by the affine map.\n"
                     "Options:\n"
                     "  -lua_input <file>   Lua file defining 'vertices', 'f' and optional 'qpts' (default tetrahedron.lua)\n\n";

/*
 * PETSc builds simplex quadratures on the biunit reference tetrahedron, whose
 * vertices are (-1,-1,-1), (1,-1,-1), (-1,1,-1), (-1,-1,1) and whose volume is
 * 4/3. A reference point (xi,eta,zeta) is mapped to the physical tetrahedron
 * with vertices p0,p1,p2,p3 by its barycentric coordinates
 *     l1 = (xi+1)/2,  l2 = (eta+1)/2,  l3 = (zeta+1)/2,  l0 = 1 - l1 - l2 - l3,
 *     p  = l0*p0 + l1*p1 + l2*p2 + l3*p3,
 * and the integral becomes
 *     \int_T f dV = |det J| * sum_q w_q f(p(xi_q)),   |det J| = |det E| / 8,
 * with E = [p1-p0, p2-p0, p3-p0].
 */

int main(int argc, char **argv)
{
    PetscErrorCode  ierr;
    lua_State       *L;                    // Lua interpreter state
    char            luafilename[PETSC_MAX_PATH_LEN] = "tetrahedron.lua"; // Lua file to load
    PetscBool       flg;
    PetscReal       P[4][3];               // The four tetrahedron vertices, read from Lua
    PetscInt        qpts = 3;              // Gauss-Jacobi points per direction (may be overridden by Lua)
    PetscQuadrature quad;                  // Quadrature rule on the reference tetrahedron
    PetscInt        dim, Nc, npoints;      // Dimension, number of components, number of points
    const PetscReal *points, *weights;     // Quadrature points and weights (owned by quad)
    PetscReal       E[3][3], detE, jac;    // Edge matrix, its determinant, and |det J|
    PetscReal       integral = 0.0;        // Accumulated value of the integral

    ierr = PetscInitialize(&argc, &argv, (char*)0, help);
    if (ierr)
        return ierr;

    // Allow the Lua file to load to be set from the command line with -lua_input
    ierr = PetscOptionsGetString(NULL, NULL, "-lua_input", luafilename, sizeof(luafilename), &flg);
    CHKERRQ(ierr);

    // Start a Lua interpreter and load the file defining the tetrahedron and the function
    if ((L = luaL_newstate()) == NULL)
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_MEM, "Error creating the Lua state.");
    luaL_openlibs(L);
    if (luaL_dofile(L, luafilename) != LUA_OK)
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Error loading the Lua file.");

    // Read the four vertices from the Lua global table 'vertices'
    lua_getglobal(L, "vertices");
    if (!lua_istable(L, -1))
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The Lua file must define a table 'vertices' with four vertices.");
    for (int i = 0; i < 4; i++)
    {
        lua_rawgeti(L, -1, i + 1);          // push vertices[i+1] (expected to be a {x,y,z} table)
        if (!lua_istable(L, -1))
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Each entry of 'vertices' must be a table {x, y, z}.");
        for (int j = 0; j < 3; j++)
        {
            lua_rawgeti(L, -1, j + 1);      // push vertices[i+1][j+1]
            P[i][j] = (PetscReal)lua_tonumber(L, -1);
            lua_pop(L, 1);
        }
        lua_pop(L, 1);                      // pop vertices[i+1]
    }
    lua_pop(L, 1);                          // pop 'vertices'

    // Optionally read the number of quadrature points per direction from the Lua global 'qpts'
    lua_getglobal(L, "qpts");
    if (lua_isnumber(L, -1))
        qpts = (PetscInt)lua_tointeger(L, -1);
    lua_pop(L, 1);
    if (qpts < 1)
        qpts = 1;

    // Build the Stroud conical quadrature on the reference tetrahedron
    ierr = PetscDTStroudConicalQuadrature(3, 1, qpts, -1.0, 1.0, &quad);
    CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(quad, &dim, &Nc, &npoints, &points, &weights);
    CHKERRQ(ierr);

    // Edge matrix E = [p1-p0, p2-p0, p3-p0]; |det J| of the affine map is |det E| / 8
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            E[j][i] = P[i + 1][j] - P[0][j];
    detE = E[0][0]*(E[1][1]*E[2][2] - E[1][2]*E[2][1])
         - E[0][1]*(E[1][0]*E[2][2] - E[1][2]*E[2][0])
         + E[0][2]*(E[1][0]*E[2][1] - E[1][1]*E[2][0]);
    jac = PetscAbsReal(detE) / 8.0;

    // For each quadrature point: map it to the physical tetrahedron, call the Lua
    // function f there, and accumulate the weighted contribution
    for (PetscInt q = 0; q < npoints; q++)
    {
        PetscReal xi  = points[3*q + 0];
        PetscReal eta = points[3*q + 1];
        PetscReal ze  = points[3*q + 2];

        PetscReal l1 = (xi  + 1.0) / 2.0;
        PetscReal l2 = (eta + 1.0) / 2.0;
        PetscReal l3 = (ze  + 1.0) / 2.0;
        PetscReal l0 = 1.0 - l1 - l2 - l3;

        PetscReal x = l0*P[0][0] + l1*P[1][0] + l2*P[2][0] + l3*P[3][0];
        PetscReal y = l0*P[0][1] + l1*P[1][1] + l2*P[2][1] + l3*P[3][1];
        PetscReal z = l0*P[0][2] + l1*P[1][2] + l2*P[2][2] + l3*P[3][2];

        // Evaluate the Lua integrand f(x, y, z)
        lua_getglobal(L, "f");
        if (!lua_isfunction(L, -1))
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The Lua file must define a function f(x, y, z).");
        lua_pushnumber(L, (double)x);
        lua_pushnumber(L, (double)y);
        lua_pushnumber(L, (double)z);
        if (lua_pcall(L, 3, 1, 0) != LUA_OK)
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Error calling the Lua function f(x, y, z).");
        integral += weights[q] * (PetscReal)lua_tonumber(L, -1);
        lua_pop(L, 1);
    }
    integral *= jac;

    ierr = PetscPrintf(PETSC_COMM_WORLD,
        "Tetrahedron read from '%s':\n"
        "  p0=(%g,%g,%g)  p1=(%g,%g,%g)  p2=(%g,%g,%g)  p3=(%g,%g,%g)\n"
        "  volume = %g\n"
        "Stroud conical quadrature: %d points per direction, %d points total, "
        "exact up to degree %d\n"
        "  integral of f over the tetrahedron = %.12g\n",
        luafilename,
        (double)P[0][0], (double)P[0][1], (double)P[0][2],
        (double)P[1][0], (double)P[1][1], (double)P[1][2],
        (double)P[2][0], (double)P[2][1], (double)P[2][2],
        (double)P[3][0], (double)P[3][1], (double)P[3][2],
        (double)(jac * 4.0 / 3.0),
        (int)qpts, (int)npoints, (int)(2*qpts - 1),
        (double)integral);
    CHKERRQ(ierr);

    ierr = PetscQuadratureDestroy(&quad);
    CHKERRQ(ierr);
    lua_close(L);
    ierr = PetscFinalize();
    return ierr;
}
