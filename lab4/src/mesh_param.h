
#ifndef MESH_PARAM
#define MESH_PARAM

#include <Eigen/Core>
#include <Eigen/Sparse>

// Debugging flags for development use
// #define DEBUG_A
// #define DEBUG_B
// #define DEBUG_1
// #define DEBUG_2
// #define DEBUG_3
// #define DEBUG_4
// #define DEBUG_5
// #define DEBUG_6
// #define DEBUG_7
// #define DEBUG_8
// #define DEBUG_9

Eigen::MatrixXd interpolate_field
(
    const Eigen::MatrixXd& V,          // Vertices of the mesh
    const Eigen::MatrixXi& F,          // Faces
    const Eigen::MatrixXi& TT,         // Adjacency triangle-triangle
    const Eigen::VectorXi& b,          // Constrained faces id
    const Eigen::MatrixXd& bc          // Cosntrained faces representative vector
);

Eigen::MatrixXd get_scalar_field
(
    const Eigen::MatrixXd& V,             // Vertices of the mesh
    const Eigen::MatrixXi& F,             // Faces
    const Eigen::MatrixXd& R,             // Vector field
    const Eigen::SparseMatrix<double>& G  // Gradient operator 
);

Eigen::MatrixXd harmonic_param
(
    const Eigen::MatrixXd& V,             // Vertices of the mesh
    const Eigen::MatrixXi& F              // Faces
);

Eigen::MatrixXd lscm_param
(
    const Eigen::MatrixXd& V,             // Vertices of the mesh
    const Eigen::MatrixXi& F              // Faces
);


#endif