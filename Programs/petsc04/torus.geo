SetFactory("OpenCASCADE");
Mesh.ScalingFactor = 0.1;

Mesh.MeshSizeMin = 0.25;
Mesh.MeshSizeMax = 0.75; 

Torus(1) = {0, 0, 0, 12, 6};
Physical Volume("Interior") = {1};
Physical Surface("Frontera") = {1};