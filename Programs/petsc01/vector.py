#!/usr/bin/python
# -*- coding: utf-8 -*-
#
""" 
Filename:   vector.py
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
from petsc4py import PETSc

data = np.loadtxt("F.txt")
x = PETSc.Vec().createWithArray(data)
viewer = PETSc.Viewer().createBinary("F.dat", mode=PETSc.Viewer.Mode.WRITE)
x.view(viewer)
