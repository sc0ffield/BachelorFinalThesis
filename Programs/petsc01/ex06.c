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
#include <math.h>
static char help[] =
    "Compute integrals over a tetrahedron using PETSc's simplex quadrature.\n"
    "The quadrature rule is built with PetscDTStroudConicalQuadrature (the Stroud\n"
    "conical rule PETSc uses on simplices) and transported from the reference\n"
    "tetrahedron to a physical one by the affine map.\n"
    "Option:  -qpts <n>   Gauss-Jacobi points per direction (default 2).\n\n";

/*
 * PETSc builds simplex quadratures on the *biunit* reference tetrahedron, whose
 * vertices are
 *     w0 = (-1,-1,-1),  w1 = (1,-1,-1),  w2 = (-1,1,-1),  w3 = (-1,-1,1),
 * and whose volume is 4/3 (equivalently, the quadrature weights add up to 4/3).
 * A point (xi,eta,zeta) of this reference tetrahedron is mapped to the physical
 * tetrahedron with vertices p0,p1,p2,p3 through its barycentric coordinates
 *     l1 = (xi+1)/2,  l2 = (eta+1)/2,  l3 = (zeta+1)/2,  l0 = 1 - l1 - l2 - l3,
 *     p  = l0*p0 + l1*p1 + l2*p2 + l3*p3.
 * Under this affine map the integral becomes
 *     \int_T f dV = |det J| * sum_q w_q f(p(xi_q)),   |det J| = |det E| / 8,
 * where E = [p1-p0, p2-p0, p3-p0] and |det E| = 6*vol(T).
 */

/* Physical tetrahedron: vertices p0,p1,p2,p3 (volume = 4). */
static const PetscReal P[4][3] = {{0, 0, 0}, {2, 0, 0}, {0, 3, 0}, {0, 0, 4}};

/* Four test integrands of increasing polynomial degree, with their exact
 * integrals over the tetrahedron above (computed analytically). */
static PetscReal f0(PetscReal x, PetscReal y, PetscReal z) { return 1.0; }            /* degree 0 */
static PetscReal f1(PetscReal x, PetscReal y, PetscReal z) { return x + 2*y + 3*z; }  /* degree 1 */
static PetscReal f2(PetscReal x, PetscReal y, PetscReal z) { return x*x; }            /* degree 2 */
static PetscReal f3(PetscReal x, PetscReal y, PetscReal z) { return x*y*z; }          /* degree 3 */

int main(int argc, char **argv)
{
    PetscErrorCode  ierr;
    PetscQuadrature quad;                 // Quadrature rule on the reference tetrahedron
    PetscInt        dim, Nc, npoints;     // Dimension, number of components, number of points
    const PetscReal *points, *weights;    // Quadrature points and weights (owned by quad)
    PetscInt        qpts = 2;             // Gauss-Jacobi points per direction
    PetscReal       E[3][3], detE, jac;   // Edge matrix, its determinant, and |det J|

    PetscReal (*f[4])(PetscReal, PetscReal, PetscReal) = {f0, f1, f2, f3};
    const char *name[4] = {"1", "x + 2y + 3z", "x^2", "x*y*z"};
    const PetscInt degree[4] = {0, 1, 2, 3};
    const PetscReal exact[4] = {4.0, 20.0, 1.6, 0.8};

    // Initialize PETSc and MPI
    ierr = PetscInitialize(&argc, &argv, (char*)0, help);
    if (ierr)
        return ierr;

    // Read the number of Gauss-Jacobi points per direction from the command line
    ierr = PetscOptionsGetInt(NULL, NULL, "-qpts", &qpts, NULL);
    CHKERRQ(ierr);

    // Build the Stroud conical quadrature on the reference tetrahedron (dim = 3,
    // 1 component, qpts points per direction, 1D interval [-1,1])
    ierr = PetscDTStroudConicalQuadrature(3, 1, qpts, -1.0, 1.0, &quad);
    CHKERRQ(ierr);

    // Retrieve the points and weights of the rule
    ierr = PetscQuadratureGetData(quad, &dim, &Nc, &npoints, &points, &weights);
    CHKERRQ(ierr);

    // Edge matrix E = [p1-p0, p2-p0, p3-p0] (columns are the edge vectors from p0)
    for (PetscInt i = 0; i < 3; i++)
        for (PetscInt j = 0; j < 3; j++)
            E[j][i] = P[i+1][j] - P[0][j];

    // Determinant of the edge matrix; |det J| of the affine map is |det E| / 8
    detE = E[0][0]*(E[1][1]*E[2][2] - E[1][2]*E[2][1])
         - E[0][1]*(E[1][0]*E[2][2] - E[1][2]*E[2][0])
         + E[0][2]*(E[1][0]*E[2][1] - E[1][1]*E[2][0]);
    jac = PetscAbsReal(detE) / 8.0;

    ierr = PetscPrintf(PETSC_COMM_WORLD,
        "Integration over the tetrahedron with vertices\n"
        "  p0=(0,0,0)  p1=(2,0,0)  p2=(0,3,0)  p3=(0,0,4),  volume = %g\n"
        "Stroud conical quadrature: %d points per direction, %d points total, "
        "exact up to degree %d\n\n", (double)(jac * 4.0 / 3.0),
        (int)qpts, (int)npoints, (int)(2*qpts - 1));
    CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
        "  %-14s %-6s %-18s %-18s %-10s\n",
        "integrand", "deg", "quadrature", "exact", "error");
    CHKERRQ(ierr);

    // For each integrand, map every reference point to the physical tetrahedron,
    // accumulate the weighted samples and scale by |det J|
    for (PetscInt k = 0; k < 4; k++)
    {
        PetscReal sum = 0.0;

        for (PetscInt q = 0; q < npoints; q++)
        {
            PetscReal xi  = points[3*q + 0];
            PetscReal eta = points[3*q + 1];
            PetscReal ze  = points[3*q + 2];

            // Barycentric coordinates on the biunit reference tetrahedron
            PetscReal l1 = (xi  + 1.0) / 2.0;
            PetscReal l2 = (eta + 1.0) / 2.0;
            PetscReal l3 = (ze  + 1.0) / 2.0;
            PetscReal l0 = 1.0 - l1 - l2 - l3;

            // Corresponding physical point p = l0*p0 + l1*p1 + l2*p2 + l3*p3
            PetscReal x = l0*P[0][0] + l1*P[1][0] + l2*P[2][0] + l3*P[3][0];
            PetscReal y = l0*P[0][1] + l1*P[1][1] + l2*P[2][1] + l3*P[3][1];
            PetscReal z = l0*P[0][2] + l1*P[1][2] + l2*P[2][2] + l3*P[3][2];

            sum += weights[q] * f[k](x, y, z);
        }
        sum *= jac;

        ierr = PetscPrintf(PETSC_COMM_WORLD,
            "  %-14s %-6d %-18.12g %-18.12g %-10.3e\n",
            name[k], (int)degree[k], (double)sum, (double)exact[k],
            (double)PetscAbsReal(sum - exact[k]));
        CHKERRQ(ierr);
    }

    ierr = PetscPrintf(PETSC_COMM_WORLD,
        "\nRe-run with '-qpts 1' to see the higher-degree integrands fail: a rule\n"
        "with one point per direction is exact only up to degree 1.\n");
    CHKERRQ(ierr);

    // Free the quadrature and finalize PETSc
    ierr = PetscQuadratureDestroy(&quad);
    CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
}
