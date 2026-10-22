/**************************************************************************************
* Filename:   ex07.c
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
/*
* Metric-based ADAPTIVE mesh refinement of the Fichera corner mesh.
*
* Unlike uniform refinement (ex01's -dm_refine, which splits every tetrahedron
* into 8), this example refines only where it is needed: near the re-entrant
* corner of the Fichera domain (the point (0.5,0.5,0.5)), where the solution of
* -div(grad u) = 1 has a gradient singularity and the discretisation error
* concentrates. It does so by prescribing a Riemannian metric field that asks
* for small elements near the corner and coarse ones away from it, and letting
* Mmg regenerate a mesh that honours it. Several passes chase the singularity to
* ever smaller scales.
*
* IMPORTANT: metric-based adaptation is performed by the external library Mmg
* (serial) or ParMmg (parallel), which PETSc calls from DMAdaptMetric(). This
* program therefore only runs on a PETSc built --with-mmg (and/or --with-parmmg).
* On a PETSc without Mmg it compiles and links, but DMAdaptMetric() returns the
* error "This method requires Mmg/ParMmg". The MareNostrum 5 toolchain, or a
* local PETSc configured with --download-mmg, provides it.
*/
#include <petscdmplex.h>
#include <petscsys.h>

static char help[] = "Metric-based adaptive refinement of the Fichera mesh towards its re-entrant corner (requires Mmg).\n\n";

/*
* Build an isotropic metric M(x) = (1/h(x)^2) I on the vertices of dm, where the
* target edge length h(x) grows linearly with the distance r to the corner:
*   h(x) = h_min + (h_max - h_min) * min(1, r / L).
* Small h (dense mesh) near the corner, large h (coarse mesh) far from it.
*/
static PetscErrorCode BuildCornerMetric(DM dm, PetscReal corner[3], PetscReal h_min, PetscReal h_max, PetscReal L, Vec *metric)
{
    DM              dmMetric;
    PetscSection    coordSection, metricSection;
    Vec             coordinates;
    const PetscScalar *coords;
    PetscScalar    *m;
    PetscInt        vStart, vEnd, v, dim;
    PetscErrorCode  ierr;

    PetscFunctionBeginUser;
    ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

    // Allocate the metric: a full dim x dim symmetric tensor at every vertex.
    ierr = DMPlexMetricCreate(dm, 0, metric); CHKERRQ(ierr);
    ierr = VecGetDM(*metric, &dmMetric); CHKERRQ(ierr);
    ierr = DMGetLocalSection(dmMetric, &metricSection); CHKERRQ(ierr);

    // Vertex coordinates.
    ierr = DMGetCoordinatesLocal(dm, &coordinates); CHKERRQ(ierr);
    ierr = DMGetCoordinateSection(dm, &coordSection); CHKERRQ(ierr);
    ierr = VecGetArrayRead(coordinates, &coords); CHKERRQ(ierr);
    ierr = VecGetArray(*metric, &m); CHKERRQ(ierr);

    ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd); CHKERRQ(ierr);   // depth 0 = vertices
    for (v = vStart; v < vEnd; ++v) {
        PetscInt  coff, moff, i, j;
        PetscReal r2 = 0.0, h, mag;

        ierr = PetscSectionGetOffset(coordSection, v, &coff); CHKERRQ(ierr);
        for (i = 0; i < dim; ++i) {
            PetscReal d = PetscRealPart(coords[coff + i]) - corner[i];
            r2 += d * d;
        }
        h   = h_min + (h_max - h_min) * PetscMin(1.0, PetscSqrtReal(r2) / L);
        mag = 1.0 / (h * h);                       // metric magnitude 1/h^2

        // Write mag*I into this vertex, respecting the storage layout: an
        // isotropic metric keeps a single scalar per vertex, a full metric the
        // whole dim x dim tensor.
        PetscInt ndof;
        ierr = PetscSectionGetOffset(metricSection, v, &moff); CHKERRQ(ierr);
        ierr = PetscSectionGetDof(metricSection, v, &ndof); CHKERRQ(ierr);
        if (ndof == 1) {
            m[moff] = mag;                          // compact isotropic storage
        } else {
            for (i = 0; i < dim; ++i)               // full dim x dim tensor storage
                for (j = 0; j < dim; ++j)
                    m[moff + i * dim + j] = (i == j) ? mag : 0.0;
        }
    }
    ierr = VecRestoreArray(*metric, &m); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(coordinates, &coords); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

// Count how many cells have their centroid within radius R of the corner,
// alongside the total. A rising near/total ratio is the signature of adaptive
// refinement: cells get spent at the singularity and saved away from it.
static PetscErrorCode CountNearCorner(DM dm, PetscReal corner[3], PetscReal R, PetscInt *nNear, PetscInt *nTot)
{
    PetscInt       cStart, cEnd, c;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);
    *nTot = cEnd - cStart;
    *nNear = 0;
    for (c = cStart; c < cEnd; ++c) {
        PetscReal cen[3];
        ierr = DMPlexComputeCellGeometryFVM(dm, c, NULL, cen, NULL); CHKERRQ(ierr);
        PetscReal d2 = (cen[0]-corner[0])*(cen[0]-corner[0]) + (cen[1]-corner[1])*(cen[1]-corner[1]) + (cen[2]-corner[2])*(cen[2]-corner[2]);
        if (d2 <= R * R) (*nNear)++;
    }
    // Sum the per-rank local counts into a global total, so the reported figures
    // describe the whole distributed mesh and not just this rank's share.
    {
        PetscInt local[2] = {*nNear, *nTot}, global[2];
        ierr = MPI_Allreduce(local, global, 2, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject)dm)); CHKERRQ(ierr);
        *nNear = global[0];
        *nTot  = global[1];
    }
    PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
    DM             dm, dmAdapt = NULL;
    DMLabel        bdLabel = NULL;
    Vec            metric;
    PetscErrorCode ierr;
    PetscMPIInt    rank;
    PetscInt       it, iters = 3, cStart, cEnd;
    PetscReal      corner[3] = {0.5, 0.5, 0.5};
    PetscReal      h_min = 0.01, h_max = 0.08, L = 0.5;
    PetscReal      targetComplexity = 15000.0;   // ~target number of vertices Mmg aims for
    char           filename[PETSC_MAX_PATH_LEN] = "../petsc07/cube3.msh";

    ierr = PetscInitialize(&argc, &argv, (char*)0, help);
    if (ierr)
        return ierr;
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);

    // Command-line controls
    ierr = PetscOptionsGetString(NULL, NULL, "-mesh_input", filename, sizeof(filename), NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-adapt_iters", &iters, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL, NULL, "-h_min", &h_min, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL, NULL, "-h_max", &h_max, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL, NULL, "-grade_length", &L, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL, NULL, "-target_complexity", &targetComplexity, NULL); CHKERRQ(ierr);

    // Read the Fichera mesh on rank 0 (interpolate = PETSC_TRUE: adaptation needs
    // the faces and edges, and it lets the "Face Sets" boundary labels survive).
    ierr = DMPlexCreateGmshFromFile(PETSC_COMM_WORLD, filename, PETSC_TRUE, &dm); CHKERRQ(ierr);

    // Distribute the mesh across the MPI ranks. DMPlexCreateGmshFromFile builds
    // the whole mesh on rank 0; without redistributing it the other ranks are
    // empty, and parallel adaptation (-dm_adaptor parmmg) then segfaults on those
    // empty ranks. In serial this is a no-op (dmDist stays NULL).
    {
        DM dmDist = NULL;
        ierr = DMPlexDistribute(dm, 0, NULL, &dmDist); CHKERRQ(ierr);
        if (dmDist) { ierr = DMDestroy(&dm); CHKERRQ(ierr); dm = dmDist; }
    }

    if (rank == 0)
        PetscPrintf(PETSC_COMM_SELF, "Adaptive refinement towards the Fichera corner (%.2f,%.2f,%.2f), h in [%.3g, %.3g]\n",
                    (double)corner[0], (double)corner[1], (double)corner[2], (double)h_min, (double)h_max);

    {
        PetscInt nNear, nTot;
        ierr = CountNearCorner(dm, corner, 0.15, &nNear, &nTot); CHKERRQ(ierr);
        PetscPrintf(PETSC_COMM_WORLD, "  initial mesh: %d tetrahedra (%d within 0.15 of the corner)\n", (int)nTot, (int)nNear);
    }

    for (it = 0; it < iters; ++it) {
        Vec determinant;
        DM  dmDet;

        // Metric controls on the current mesh: isotropic storage, element-size
        // clamps, and the target complexity Mmg should aim for. They must be
        // (re)set on every mesh, since each pass produces a brand-new DM.
        ierr = DMPlexMetricSetIsotropic(dm, PETSC_TRUE); CHKERRQ(ierr);
        ierr = DMPlexMetricSetMinimumMagnitude(dm, 1.0 / (h_max * h_max)); CHKERRQ(ierr);
        ierr = DMPlexMetricSetMaximumMagnitude(dm, 1.0 / (h_min * h_min)); CHKERRQ(ierr);
        ierr = DMPlexMetricSetTargetComplexity(dm, targetComplexity); CHKERRQ(ierr);
        ierr = DMPlexMetricSetFromOptions(dm); CHKERRQ(ierr);

        // Raw metric: fine near the corner, coarse away from it (relative density).
        ierr = BuildCornerMetric(dm, corner, h_min, h_max, L, &metric); CHKERRQ(ierr);

        // Make it symmetric positive-definite and rescale it to the target
        // complexity, clamping element sizes. Normalisation is what keeps Mmg
        // from asking for an unbounded number of elements ("lack of memory").
        ierr = DMPlexMetricDeterminantCreate(dm, 0, &determinant, &dmDet); CHKERRQ(ierr);
        ierr = DMPlexMetricEnforceSPD(dm, metric, PETSC_TRUE, PETSC_FALSE, metric, determinant); CHKERRQ(ierr);
        ierr = DMPlexMetricNormalize(dm, metric, PETSC_TRUE, PETSC_FALSE, metric, determinant); CHKERRQ(ierr);

        // Keep the Gmsh boundary tags fixed through the remeshing.
        ierr = DMGetLabel(dm, "Face Sets", &bdLabel); CHKERRQ(ierr);

        // Hand the metric to Mmg. It returns a brand-new mesh honouring it,
        // dense near the corner and coarse elsewhere -- NOT a uniform x8 split.
        ierr = DMAdaptMetric(dm, metric, bdLabel, NULL, &dmAdapt); CHKERRQ(ierr);

        ierr = VecDestroy(&metric); CHKERRQ(ierr);
        ierr = VecDestroy(&determinant); CHKERRQ(ierr);
        ierr = DMDestroy(&dmDet); CHKERRQ(ierr);
        ierr = DMDestroy(&dm); CHKERRQ(ierr);
        dm   = dmAdapt;

        {
            PetscInt nNear, nTot;
            ierr = CountNearCorner(dm, corner, 0.15, &nNear, &nTot); CHKERRQ(ierr);
            PetscPrintf(PETSC_COMM_WORLD, "  pass %d (target complexity %.0f): %d tetrahedra (%d within 0.15 of the corner)\n",
                        (int)it + 1, (double)targetComplexity, (int)nTot, (int)nNear);
        }

        // Chase the singularity to a finer scale on the next pass.
        targetComplexity *= 2.0;
    }

    // Write the final adapted mesh for inspection in ParaView.
    {
        PetscViewer viewer;
        ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD, "fichera_adapted.vtu", FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);
        ierr = DMView(dm, viewer); CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    }

    ierr = DMDestroy(&dm); CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
}
