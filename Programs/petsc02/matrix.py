#!/usr/bin/python
# -*- coding: utf-8 -*-
#
""" 
Filename:   matrix.py
Author:     Rafel Amer (rafel.amer AT upc.edu)
Copyright:  Rafel Amer 2026
Disclaimer: This code is presented "as is" and it has been written for
educational purposes.

License:    This library  is free software; you can redistribute it and/or
modify it under the terms of either:

1 the GNU Lesser General Public License as published by the Free
Software Foundation; either version 3 of the License, or (at your
option) any later version.

or

2 the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option)
any later version.

	      See https://www.gnu.org/licenses/
"""
import numpy as np
import scipy.sparse as sparse
import sys
from petsc4py import PETSc

print("1. Carregant el fitxer de text K.txt...")
dades = np.loadtxt('K.txt')

rows = dades[:, 0].astype(np.int32)
cols = dades[:, 1].astype(np.int32)
vals = dades[:, 2].astype(np.float64)

nrows = int(rows.max() + 1)
ncols = int(cols.max() + 1)

print(f"2. Creant matriu SciPy ({nrows}x{ncols})...")
matriu_coo = sparse.coo_matrix((vals, (rows, cols)), shape=(nrows, ncols), dtype=np.float64)
matriu_csr = matriu_coo.tocsr()
matriu_csr.sum_duplicates()

A = PETSc.Mat().createAIJ(size=matriu_csr.shape, csr=(matriu_csr.indptr, matriu_csr.indices, matriu_csr.data))
viewer = PETSc.Viewer().createBinary('K.dat', 'w')
A.view(viewer)
print("Èxit! El fitxer K.dat ha estat generat de manera nativa per PETSc.")
