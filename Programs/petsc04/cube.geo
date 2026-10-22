SetFactory("OpenCASCADE");
Mesh.ScalingFactor = 0.1;

Mesh.MeshSizeMin = 0.5;
Mesh.MeshSizeMax = 1.0;

// A 30x30x30 cube with a radius-10 cylindrical hole drilled through its center along X,
// built directly with the OpenCASCADE kernel (equivalent to the old cube.scad model).
Box(1) = {0, 0, 0, 30, 30, 30};
Cylinder(2) = {-5, 15, 15, 40, 0, 0, 10};
BooleanDifference(3) = { Volume{1}; Delete; }{ Volume{2}; Delete; };

Physical Volume("Interior")      = {3};
Physical Surface("Cara_X_Min")   = {1};
Physical Surface("Cara_Y_Min")   = {2};
Physical Surface("Cara_Z_Max")   = {3};
Physical Surface("Cara_Z_min")   = {4};
Physical Surface("Cara_Y_Max")   = {5};
Physical Surface("Cilindre_Int") = {6};
Physical Surface("Cara_X_Max")   = {7};
