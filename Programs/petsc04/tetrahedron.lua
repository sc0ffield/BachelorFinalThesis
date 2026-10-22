-- Definition of a tetrahedron and of a function to integrate over it.
-- This file is read by ex06.c, which computes  \int_T f(x,y,z) dV  using
-- PETSc's simplex quadrature. Both the tetrahedron and the integrand are
-- defined here, so the domain and the function can be changed without
-- recompiling the C program.

-- The four vertices p0, p1, p2, p3 of the tetrahedron T
vertices = {
    { 0.0, 0.0, 0.0 },
    { 2.0, 0.0, 0.0 },
    { 0.0, 3.0, 0.0 },
    { 0.0, 0.0, 4.0 },
}

-- Number of Gauss-Jacobi points per direction for the Stroud conical rule.
-- The rule is exact for polynomials of degree up to 2*qpts - 1; lower it to
-- see the quadrature error appear for high-degree integrands.
qpts = 3

-- The integrand f(x, y, z). For the tetrahedron above the exact value of
-- \int_T x*y*z dV is 0.8, which the program reproduces to machine precision.
function f(x, y, z)
    return x * y * z
end
