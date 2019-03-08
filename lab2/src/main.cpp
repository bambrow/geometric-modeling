#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
/*** insert any necessary libigl headers here ***/
#include <igl/per_face_normals.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <igl/copyleft/marching_cubes.h>
#include <deque>
#include <algorithm>
#include <igl/Timer.h>
#include <igl/facet_components.h>

using namespace std;
using Viewer = igl::opengl::glfw::Viewer;

// Input: imported points, #P x3
Eigen::MatrixXd P;

// Input: imported normals, #P x3
Eigen::MatrixXd N;

// Intermediate result: constrained points, #C x3
Eigen::MatrixXd constrained_points;

// Intermediate result: implicit function values at constrained points, #C x1
Eigen::VectorXd constrained_values;

// Parameter: degree of the polynomial
unsigned int polyDegree = 2;

// Parameter: Wendland weight function radius (make this relative to the size of the mesh)
double wendlandRadius = 0.1;

// Parameter: grid resolution
unsigned int resolution = 20;

// Intermediate result: grid points, at which the imlicit function will be evaluated, #G x3
Eigen::MatrixXd grid_points;

// Intermediate result: implicit function values at the grid points, #G x1
Eigen::VectorXd grid_values;

// Intermediate result: grid point colors, for display, #G x3
Eigen::MatrixXd grid_colors;

// Intermediate result: grid lines, for display, #L x6 (each row contains
// starting and ending point of line segment)
Eigen::MatrixXd grid_lines;

// Output: vertex array, #V x3
Eigen::MatrixXd V;

// Output: face array, #F x3
Eigen::MatrixXi F;

// Output: face normals of the reconstructed mesh, #F x3
Eigen::MatrixXd FN;

// Additional global variables
double epsilon = 0.0;
Eigen::RowVector3d si_upper_bound;
Eigen::RowVector3d si_lower_bound;
double si_dim = 0.0;
int si_axis_x = 0;
int si_axis_y = 0;
int si_axis_z = 0;
map<int, vector<int>> si_map;
map<int, vector<int>> si_constrained_map;

// Functions
void createGrid();
void evaluateImplicitFunc();
void getLines();
bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers);

// Additional functions
bool isClosest(Eigen::RowVector3d& p1, Eigen::RowVector3d& p2);
bool isClosestBrutal(Eigen::RowVector3d& p1, Eigen::RowVector3d& p2);
double getDistance(Eigen::RowVector3d& p1, Eigen::RowVector3d& p2);
int calculateSpatialIndex(int xi, int yi, int zi);
int getSpatialIndex(Eigen::RowVector3d& p);
vector<int> getNeighborCells(int& cell_id);
Eigen::RowVector3d getClosestPoint(Eigen::RowVector3d& p);
bool isSamePoint(Eigen::RowVector3d& p1, Eigen::RowVector3d& p2);
vector<int> getPointsWithinDistance(Eigen::RowVector3d& p, double h);
vector<int> getPointsWithinDistanceBrutal(Eigen::RowVector3d& p, double h);
double wendlandWeights(double r, double h);

// Creates a grid_points array for the simple sphere example. The points are
// stacked into a single matrix, ordered first in the x, then in the y and
// then in the z direction. If you find it necessary, replace this with your own
// function for creating the grid.
void createGrid() {
    grid_points.resize(0, 3);
    grid_colors.resize(0, 3);
    grid_lines. resize(0, 6);
    grid_values.resize(0);
    V. resize(0, 3);
    F. resize(0, 3);
    FN.resize(0, 3);

    // Grid bounds: axis-aligned bounding box
    Eigen::RowVector3d bb_min, bb_max;
    bb_min = P.colwise().minCoeff();
    bb_max = P.colwise().maxCoeff();

    // Use PCA to align axes
    Eigen::RowVector3d bb_mean = P.colwise().mean();
    Eigen::MatrixXd centered = P.rowwise() - bb_mean;
    Eigen::MatrixXd cov = centered.adjoint() * centered;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(cov);
    Eigen::MatrixXd eig_vectors = eig.eigenvectors();
    Eigen::Matrix3d eig_dim = eig_vectors.rightCols(3);
    Eigen::MatrixXd pca_points = P * eig_dim;
    Eigen::RowVector3d pca_bb_min = pca_points.colwise().minCoeff();
    Eigen::RowVector3d pca_bb_max = pca_points.colwise().maxCoeff();
    Eigen::RowVector3d pca_bb_dim = pca_bb_max - pca_bb_min;
    
    // Enlarge the bouding box slightly according to diagonal
    if (epsilon == 0.0) {
        Eigen::RowVector3d original_dim = bb_max - bb_min;
        double bb_diagonal = sqrt(original_dim[0] * original_dim[0] + original_dim[1] * original_dim[1] + original_dim[2] * original_dim[2]);
        epsilon = 0.01 * bb_diagonal;
    }
    Eigen::RowVector3d bb_dim = bb_max - bb_min;
    // Eigen::RowVector3d grid_lower_bound = bb_min - 0.1 * bb_dim;
    // Eigen::RowVector3d grid_upper_bound = bb_max + 0.1 * bb_dim;
    Eigen::RowVector3d pca_lower_bound = pca_bb_min - 0.1 * pca_bb_dim;
    Eigen::RowVector3d pca_upper_bound = pca_bb_max + 0.1 * pca_bb_dim;

    // Bounding box dimensions
    // Eigen::RowVector3d dim = bb_max - bb_min;
    // Eigen::RowVector3d dim = grid_upper_bound - grid_lower_bound;
    Eigen::RowVector3d dim = pca_upper_bound - pca_lower_bound;

    // Grid spacing
    const double dx = dim[0] / (double)(resolution - 1);
    const double dy = dim[1] / (double)(resolution - 1);
    const double dz = dim[2] / (double)(resolution - 1);
    // 3D positions of the grid points -- see slides or marching_cubes.h for ordering
    grid_points.resize(resolution * resolution * resolution, 3);
    // Create each gridpoint
    for (unsigned int x = 0; x < resolution; ++x) {
        for (unsigned int y = 0; y < resolution; ++y) {
            for (unsigned int z = 0; z < resolution; ++z) {
                // Linear index of the point at (x,y,z)
                int index = x + resolution * (y + resolution * z);
                // The library expects the points to be ordered lexicographically
                //   by their (z,y,x) grid index
                // 3D point at (x,y,z)

                // grid_points.row(index) = bb_min + Eigen::RowVector3d(x * dx, y * dy, z * dz);
                // grid_points.row(index) = grid_lower_bound + Eigen::RowVector3d(x * dx, y * dy, z * dz);
                grid_points.row(index) = pca_lower_bound + Eigen::RowVector3d(x * dx, y * dy, z * dz);
            }
        }
    }
    // transform back
    grid_points = grid_points * eig_dim.inverse();
}

// Function for explicitly evaluating the implicit function for a sphere of
// radius r centered at c : f(p) = ||p-c|| - r, where p = (x,y,z).
// This will NOT produce valid results for any mesh other than the given
// sphere.
// Replace this with your own function for evaluating the implicit function
// values at the grid points using MLS
void evaluateImplicitFunc() {

    int nnodes = grid_points.rows();
    grid_values.resize(nnodes);
    // cout << "nnodes: " << nnodes << endl;

    // Add timer and time indicators
    igl::Timer total_timer;
    double si_total_time;
    double linear_solving_total_time;

    // Build matrix B; it is independent from x
    total_timer.start();
    int k; // polynomial combinations
    int n3; // number of constrained points
    if (polyDegree <= 0) {
        k = 1; // 1
    } else if (polyDegree == 1) {
        k = 4; // 1, x, y, z
    } else {
        k = 10; // 1, x, y, z, xy, xz, yz, x^2, y^2, z^2
    }
    n3 = constrained_points.rows();
    Eigen::MatrixXd B;
    B.resize(n3, k); // B is 3n * k
    for (int i = 0; i < n3; i++) {
        Eigen::RowVector3d cp = constrained_points.row(i);
        B(i, 0) = 1; // 1
        if (k >= 4) {
            B(i, 1) = cp[0]; // x
            B(i, 2) = cp[1]; // y
            B(i, 3) = cp[2]; // z
        }
        if (k >= 10) {
            B(i, 4) = cp[0] * cp[1]; // xy
            B(i, 5) = cp[0] * cp[2]; // xz
            B(i, 6) = cp[1] * cp[2]; // yz
            B(i, 7) = cp[0] * cp[0]; // x^2
            B(i, 8) = cp[1] * cp[1]; // y^2
            B(i, 9) = cp[2] * cp[2]; // z^2
        }
    }
    Eigen::MatrixXd BT = B.transpose(); // BT is k * 3n

    cout << "Dimensions of Matrix B: " << B.rows() << " * " << B.cols() << endl;
    cout << "Polynomial degree = " << polyDegree << "; Wendland radius = " << wendlandRadius << endl;

    // cout << B << endl;

    for (int i = 0; i < nnodes; i++) {
        Eigen::RowVector3d gp = grid_points.row(i);
        // Build matrix W(x)
        Eigen::SparseMatrix<double> W;
        W.resize(n3, n3); // W is 3n * 3n
        
        // Add timer
        igl::Timer si_timer;
        si_timer.start();
        vector<int> points_nonzero = getPointsWithinDistance(gp, wendlandRadius);
        for (int point_nonzero : points_nonzero) {
            Eigen::RowVector3d cp = constrained_points.row(point_nonzero);
            W.insert(point_nonzero, point_nonzero) = wendlandWeights((gp - cp).norm(), wendlandRadius);
        }
        si_timer.stop();
        si_total_time += si_timer.getElapsedTime();
        if (points_nonzero.size() == 0) {
            grid_values(i) = std::numeric_limits<double>::max(); // no constrained point within Wendland radius
        }
        
        /*
        // Commented: brute force approach
        for (int j = 0; j < n3; j++) {
            Eigen::RowVector3d cp = constrained_points.row(j);
            W(j, j) = wendlandWeights((gp - cp).norm(), wendlandRadius);
        }
        */

        // Add timer
        igl::Timer linear_solving_timer;
        linear_solving_timer.start();
        // Prepare for linear solving
        Eigen::MatrixXd left_matrix = (BT * W) * B; // left: k * k 
        Eigen::MatrixXd right_matrix = (BT * W) * constrained_values; // right: k * 1
        // cout << "left/right complete for " << i << endl;
        // Solve a(x): k * 1
        Eigen::VectorXd ax = left_matrix.colPivHouseholderQr().solve(right_matrix);
        // Eigen::VectorXd ax = left_matrix.inverse() * right_matrix;
        // cout << "solving ax complete for " << i << endl;
        linear_solving_timer.stop();
        linear_solving_total_time += linear_solving_timer.getElapsedTime();

        // Prepare bx: 1 * k
        Eigen::RowVectorXd bx;
        bx.resize(k);
        bx(0) = 1; // 1
        if (k >= 4) {
            bx(1) = gp[0]; // x
            bx(2) = gp[1]; // y
            bx(3) = gp[2]; // z
        }
        if (k >= 10) {
            bx(4) = gp[0] * gp[1]; // xy
            bx(5) = gp[0] * gp[2]; // xz
            bx(6) = gp[1] * gp[2]; // yz
            bx(7) = gp[0] * gp[0]; // x^2
            bx(8) = gp[1] * gp[1]; // y^2
            bx(9) = gp[2] * gp[2]; // z^2
        }
        // Finally, solve the result fx
        double fx = bx * ax;
        grid_values(i) = fx;
        if (i > 0 && i % 1000 == 0) {
            cout << "  [Progress Report] Calculating f(x) complete for grid point " << i << endl;
        }
    }

    cout << "Time spent on finding neighbors in MLS: " << si_total_time << endl;
    cout << "Time spent on solving linear system in MLS: " << linear_solving_total_time << endl;
    total_timer.stop();
    cout << "Total time spent on implicit function evaluation: " << total_timer.getElapsedTime() << endl;

    // Old version (sphere generation) below
    /*
    // Sphere center
    auto bb_min = grid_points.colwise().minCoeff().eval();
    auto bb_max = grid_points.colwise().maxCoeff().eval();
    Eigen::RowVector3d center = 0.5 * (bb_min + bb_max);

    double radius = 0.5 * (bb_max - bb_min).minCoeff();

    // Scalar values of the grid points (the implicit function values)
    grid_values.resize(resolution * resolution * resolution);

    // Evaluate sphere's signed distance function at each gridpoint.
    for (unsigned int x = 0; x < resolution; ++x) {
        for (unsigned int y = 0; y < resolution; ++y) {
            for (unsigned int z = 0; z < resolution; ++z) {
                // Linear index of the point at (x,y,z)
                int index = x + resolution * (y + resolution * z);

                // Value at (x,y,z) = implicit function for the sphere
                grid_values[index] = (grid_points.row(index) - center).norm() - radius;
            }
        }
    }
    */
}

double wendlandWeights(double r, double h) {
    if (r >= h) return 0;
    double rh = r / h;
    return pow(1 - rh, 4.0) * (4 * rh + 1);
}

// Code to display the grid lines given a grid structure of the given form.
// Assumes grid_points have been correctly assigned
// Replace with your own code for displaying lines if need be.
void getLines() {
    int nnodes = grid_points.rows();
    grid_lines.resize(3 * nnodes, 6);
    int numLines = 0;

    for (unsigned int x = 0; x < resolution; ++x) {
        for (unsigned int y = 0; y < resolution; ++y) {
            for (unsigned int z = 0; z < resolution; ++z) {
                int index = x + resolution * (y + resolution * z);
                if (x < resolution - 1) {
                    int index1 = (x + 1) + y * resolution + z * resolution * resolution;
                    grid_lines.row(numLines++) << grid_points.row(index), grid_points.row(index1);
                }
                if (y < resolution - 1) {
                    int index1 = x + (y + 1) * resolution + z * resolution * resolution;
                    grid_lines.row(numLines++) << grid_points.row(index), grid_points.row(index1);
                }
                if (z < resolution - 1) {
                    int index1 = x + y * resolution + (z + 1) * resolution * resolution;
                    grid_lines.row(numLines++) << grid_points.row(index), grid_points.row(index1);
                }
            }
        }
    }

    grid_lines.conservativeResize(numLines, Eigen::NoChange);
}

bool callback_key_down(Viewer &viewer, unsigned char key, int modifiers) {
    if (key == '1') {
        // Show imported points
        viewer.data().clear();
        viewer.core.align_camera_center(P);
        viewer.data().point_size = 11;
        viewer.data().add_points(P, Eigen::RowVector3d(0,0,0));
    }

    if (key == '2') {
        // Show all constraints
        viewer.data().clear();
        viewer.core.align_camera_center(P);
        // Add your code for computing auxiliary constraint points here

        // Add timers
        igl::Timer si_creation_timer;
        igl::Timer si_constrained_timer;

        // Get the initial epsilon
        Eigen::RowVector3d bb_min, bb_max;
        bb_min = P.colwise().minCoeff();
        bb_max = P.colwise().maxCoeff();
        if (epsilon == 0.0) {
            Eigen::RowVector3d original_dim = bb_max - bb_min;
            double bb_diagonal = sqrt(original_dim[0] * original_dim[0] + original_dim[1] * original_dim[1] + original_dim[2] * original_dim[2]);
            epsilon = 0.01 * bb_diagonal;
        }

        // Prepare data structures for spatial index; enlarge the bounding box slightly
        si_creation_timer.start();
        Eigen::RowVector3d bb_dim = bb_max - bb_min;
        si_lower_bound = bb_min - 0.1 * bb_dim;
        si_upper_bound = bb_max + 0.1 * bb_dim;
        int si_original_num = (int) pow(P.rows(), 1.0 / 2);
        Eigen::RowVector3d total_dim = si_upper_bound - si_lower_bound;
        assert(total_dim[0] > 0 && total_dim[1] > 0 && total_dim[2] > 0);
        double total_vol = total_dim[0] * total_dim[1] * total_dim[2];
        si_dim = pow(total_vol / si_original_num, 1.0 / 3);
        si_axis_x = (int) ceil(total_dim[0] / si_dim);
        si_axis_y = (int) ceil(total_dim[1] / si_dim);
        si_axis_z = (int) ceil(total_dim[2] / si_dim);

        // Prepare grid map
        si_map.clear();
        for (int i = 0; i < P.rows(); i++) {
            Eigen::RowVector3d p = P.row(i);
            int cell_id = getSpatialIndex(p);
            si_map[cell_id].push_back(i);
        }
        si_creation_timer.stop();

        // Suggest a Wendland radius here
        wendlandRadius = si_dim * 1.2;

        cout << "Total number of points: " << P.rows() << endl;
        cout << "Bounding box dimensions: " << total_dim << endl;
        cout << "Bounding box volume: " << total_vol << endl;
        cout << "Intended cell dimensions for spatial index: " << si_dim << " with (x, y, z) = " << si_axis_x << ", " << si_axis_y << ", " << si_axis_z << endl;
        cout << "Size of spatial index map: " << si_map.size() << endl; 
        cout << "Suggested Wendland radius: " << wendlandRadius << endl;

        /*
        // Test: contents in si_map
        for (auto it0 = si_map.begin(); it0 != si_map.end(); it0++) {
            int k0 = it0 -> first;
            vector<int> v0 = it0 -> second;
            cout << "[" << k0 << ", ";
            for (int v1 : v0) {
                cout << v1 << " ";
            }
            cout << "]" << endl;
        }
        */
        /*
        // Test: neighbor cells
        for (auto it1 = si_map.begin(); it1 != si_map.end(); it1++) {
            int k1 = it1 -> first;
            vector<int> v1 = it1 -> second;
            cout << k1 << ": ";
            for (int v2 : v1) {
                cout << v2 << " ";
            }
            cout << endl << "   | ";
            vector<int> nbc = getNeighborCells(k1);
            for (int nbci : nbc) {
                cout << nbci << "|";
            }
            cout << endl;
        }
        */

        // Compute constraints
        si_constrained_timer.start();
        constrained_points.resize(P.rows() * 3, P.cols());
        constrained_values.resize(P.rows() * 3);

        // Initial assignments
        for (int i = 0; i < P.rows(); i++) {
            constrained_points.row(i) = P.row(i);
            constrained_values[i] = 0.0;
        }

        // Check epsilon
        assert(epsilon > 0);

        // Compute inward
        for (int i = 0; i < P.rows(); i++) {
            // cout << "---" << i << "---" << endl;
            int j = i + P.rows() * 2;
            Eigen::RowVector3d p1 = P.row(i);
            Eigen::RowVector3d p2;
            double current_epsilon = epsilon;
            p2 = p1 - current_epsilon * N.row(i);
            // cout << p2 << endl;
            while (!isClosest(p2, p1)) {
                current_epsilon /= 2;
                p2 = p1 - current_epsilon * N.row(i);
                // cout << p2 << endl;
            }
            constrained_points.row(j) = p2;
            constrained_values[j] = -current_epsilon;
        }

        // Compute outward
        for (int i = 0; i < P.rows(); i++) {
            int j = i + P.rows();
            Eigen::RowVector3d p1 = P.row(i);
            Eigen::RowVector3d p2;
            double current_epsilon = epsilon;
            p2 = p1 + current_epsilon * N.row(i);
            while (!isClosest(p2, p1)) {
                current_epsilon /= 2;
                p2 = p1 + current_epsilon * N.row(i);
            }
            constrained_points.row(j) = p2;
            constrained_values[j] = current_epsilon;
        }
        si_constrained_timer.stop();
        
        cout << "Time spent on creating spatial index: " << si_creation_timer.getElapsedTime() << endl;
        cout << "Time spent on computing constrained points: " << si_constrained_timer.getElapsedTime() << endl;

        // Add code for displaying all points, as above
        viewer.data().point_size = 3;
        viewer.data().add_points(P, Eigen::RowVector3d(0, 0, 255));
        viewer.data().add_points(constrained_points.middleRows(P.rows(), P.rows()), Eigen::RowVector3d(255, 0, 0));
        viewer.data().add_points(constrained_points.bottomRows(P.rows()), Eigen::RowVector3d(0, 255, 0));
    }

    if (key == '3') {
        // Show grid points with colored nodes and connected with lines
        viewer.data().clear();
        // viewer.core.align_camera_center(P);
        viewer.core.align_camera_center(grid_points);
        // Add code for creating a grid
        // Add your code for evaluating the implicit function at the grid points
        // Add code for displaying points and lines
        // You can use the following example:   

        /*** begin: sphere example, replace (at least partially) with your code ***/

        // First, another spatial index for constrained points is needed
        si_constrained_map.clear();
        for (int i = 0; i < constrained_points.rows(); i++) {
            Eigen::RowVector3d p = constrained_points.row(i);
            int cell_id = getSpatialIndex(p);
            si_constrained_map[cell_id].push_back(i);
        }

        /*
        // Test: points within distance
        assert(si_dim > 0);
        Eigen::RowVector3d point_within_test = P.row(0);
        cout << "0: " << point_within_test << " in cell: " << getSpatialIndex(point_within_test) << endl << "----with distance test----" << endl;
        for (double db = 0.1; db <= 4; db += 0.1) {
            vector<int> points_within = getPointsWithinDistance(point_within_test, si_dim * db);
            vector<int> points_within_brutal = getPointsWithinDistanceBrutal(point_within_test, si_dim * db);
            cout << "total: " << points_within.size() << " compared to brutal: " << points_within_brutal.size() << endl;
        }
        */    

        // Make grid
        createGrid();

        // Evaluate implicit function
        evaluateImplicitFunc();

        // get grid lines
        getLines();

        // Code for coloring and displaying the grid points and lines
        // Assumes that grid_values and grid_points have been correctly assigned.
        grid_colors.setZero(grid_points.rows(), 3);

        // Build color map
        for (int i = 0; i < grid_points.rows(); ++i) {
            double value = grid_values(i);
            if (value < 0) {
                grid_colors(i, 1) = 1;
            }
            else {
                if (value > 0)
                    grid_colors(i, 0) = 1;
            }
        }

        // Draw lines and points
        viewer.data().point_size = 8;
        viewer.data().add_points(grid_points, grid_colors);
        viewer.data().add_edges(grid_lines.block(0, 0, grid_lines.rows(), 3),
                              grid_lines.block(0, 3, grid_lines.rows(), 3),
                              Eigen::RowVector3d(0.8, 0.8, 0.8));
        /*** end: sphere example ***/
    }

    if (key == '4') {
        // Show reconstructed mesh
        viewer.data().clear();
        // Code for computing the mesh (V,F) from grid_points and grid_values
        if ((grid_points.rows() == 0) || (grid_values.rows() == 0)) {
            cerr << "Not enough data for Marching Cubes !" << endl;
            return true;
        }
        // Run marching cubes
        igl::copyleft::marching_cubes(grid_values, grid_points, resolution, resolution, resolution, V, F);
        if (V.rows() == 0) {
            cerr << "Marching Cubes failed!" << endl;
            return true;
        }

        // Extra: filtering
        Eigen::VectorXi C;
        igl::facet_components(F, C);
        map<int, int> occurrence_map;
        for (int i = 0; i < C.rows(); i++) {
            int connected_index = C[i];
            occurrence_map[connected_index]++;
        }
        int occurrence_max_key;
        int occurrence_max_count = 0;
        for (auto it1 = occurrence_map.begin(); it1 != occurrence_map.end(); it1++) {
            int current_key = it1 -> first;
            int current_count = it1 -> second;
            if (current_count > occurrence_max_count) {
                occurrence_max_key = current_key;
                occurrence_max_count = current_count;
            }
        }
        Eigen::MatrixXi filtered;
        filtered.resize(0, 3);
        for (int i = 0; i < C.rows(); i++) {
            if (C[i] == occurrence_max_key) {
                Eigen::RowVector3i current_row = F.row(i);
                filtered.conservativeResize(filtered.rows() + 1, 3);
                filtered.row(filtered.rows() - 1) = current_row;
            }
        }
        F = filtered;

        igl::per_face_normals(V, F, FN);
        viewer.data().set_mesh(V, F);
        viewer.data().show_lines = true;
        viewer.data().show_faces = true;
        viewer.data().set_normals(FN);
    }

    igl::writeOFF("output.off", V, F);

    return true;
}

Eigen::RowVector3d getClosestPoint(Eigen::RowVector3d& p) {

    int pid = getSpatialIndex(p);
    set<int> si_checked;
    list<int> si_current_level;
    set<int> si_next_level;
    Eigen::RowVector3d candidate;
    bool candidate_found = false;
    double min_distance = std::numeric_limits<double>::max();
    int current_layer = 0;

    si_current_level.push_back(pid);

    while (si_current_level.size() > 0) {
        int index_current = si_current_level.front();
        si_checked.insert(index_current);
        auto iter_search = si_map.find(index_current);
        if (iter_search != si_map.end()) {
            vector<int> points_to_check = iter_search -> second;
            for (int point_id : points_to_check) {
                Eigen::RowVector3d point_to_check = P.row(point_id);
                double current_distance = getDistance(p, point_to_check);
                if (current_distance < min_distance) {
                    min_distance = current_distance;
                    candidate = point_to_check;
                    if (!candidate_found) {
                        candidate_found = true;
                        int search_layer = (int) ceil(min_distance / si_dim);
                        while (current_layer <= search_layer) {
                            for (int gc_si : si_checked) {
                                vector<int> gc_neighbors = getNeighborCells(gc_si);
                                for (int nb : gc_neighbors) {
                                    if (si_checked.find(nb) == si_checked.end() 
                                    && std::find(si_current_level.begin(), si_current_level.end(), nb) == si_current_level.end()) {
                                        si_next_level.insert(nb);
                                    }
                                }
                            }
                            for (int gc_si : si_current_level) {
                                vector<int> gc_neighbors = getNeighborCells(gc_si);
                                for (int nb : gc_neighbors) {
                                    if (si_checked.find(nb) == si_checked.end() 
                                    && std::find(si_current_level.begin(), si_current_level.end(), nb) == si_current_level.end()) {
                                        si_next_level.insert(nb);
                                    }
                                }
                            }
                            current_layer++;
                        }
                    } 
                } 
            }
        }
        si_current_level.pop_front();
        if (si_current_level.size() == 0 && !candidate_found) {
            for (int gc_si : si_checked) {
                vector<int> gc_neighbors = getNeighborCells(gc_si);
                for (int nb : gc_neighbors) {
                    if (si_checked.find(nb) == si_checked.end() 
                      && std::find(si_current_level.begin(), si_current_level.end(), nb) == si_current_level.end()) {
                        si_next_level.insert(nb);
                    }
                }
            }
            current_layer++;
        }
        if (si_current_level.size() == 0 && si_next_level.size() > 0) {
            si_current_level.insert(si_current_level.end(), si_next_level.begin(), si_next_level.end());
            si_next_level.clear();
        }
    }

    return candidate;

}

bool isSamePoint(Eigen::RowVector3d& p1, Eigen::RowVector3d& p2) {
    double eps = 0.01 * epsilon;
    return abs(p1[0] - p2[0]) < eps && abs(p1[1] - p2[1]) < eps && abs(p1[2] - p2[2]) < eps;
}

bool isClosest(Eigen::RowVector3d& p1, Eigen::RowVector3d& p2) {
    
    Eigen::RowVector3d point_to_compare = getClosestPoint(p1);
    // cout << "comp: " << point_to_compare << " | " << getSpatialIndex(point_to_compare) << " with: " << p2 << " | " << getSpatialIndex(p2) << endl;
    return isSamePoint(point_to_compare, p2);

}

bool isClosestBrutal(Eigen::RowVector3d& p1, Eigen::RowVector3d& p2) {
    for (int i = 0; i < P.rows(); i++) {
        Eigen::RowVector3d current_point = P.row(i);
        if (getDistance(p1, current_point) < getDistance(p1, p2)) {
            return false;
        }
    }
    return true;
}

double getDistance(Eigen::RowVector3d& p1, Eigen::RowVector3d& p2) {
    double xd = p1[0] - p2[0];
    double yd = p1[1] - p2[1];
    double zd = p1[2] - p2[2];
    return sqrt(xd * xd + yd * yd + zd * zd);
}

int calculateSpatialIndex(int xi, int yi, int zi) {
    int si_axis_max = max(max(si_axis_x, si_axis_y), si_axis_z);
    return ((xi * si_axis_max) + yi) * si_axis_max + zi;
}

int getSpatialIndex(Eigen::RowVector3d& p) {
    assert(si_upper_bound.size() > 0 && si_lower_bound.size() > 0 && si_dim > 0);
    assert(si_axis_x > 0 && si_axis_y > 0 && si_axis_z > 0);
    assert(p[0] >= si_lower_bound[0] && p[1] >= si_lower_bound[1] && p[2] >= si_lower_bound[2]);
    assert(p[0] <= si_upper_bound[0] && p[1] <= si_upper_bound[1] && p[2] <= si_upper_bound[2]);
    int xi = (int) ((p[0] - si_lower_bound[0]) / si_dim);
    if (xi == si_axis_x) xi--;
    int yi = (int) ((p[1] - si_lower_bound[1]) / si_dim);
    if (yi == si_axis_y) yi--;
    int zi = (int) ((p[2] - si_lower_bound[2]) / si_dim);
    if (zi == si_axis_z) zi--;
    return calculateSpatialIndex(xi, yi, zi);
}

vector<int> getNeighborCells(int& cell_id) {
    assert(si_upper_bound.size() > 0 && si_lower_bound.size() > 0 && si_dim > 0);
    assert(si_axis_x > 0 && si_axis_y > 0 && si_axis_z > 0);
    int si_axis_max = max(max(si_axis_x, si_axis_y), si_axis_z);
    assert(cell_id >= 0 && cell_id < si_axis_max * si_axis_max * si_axis_max);
    vector<int> neighbors;
    int zi = cell_id % si_axis_max;
    int yi = ((cell_id - zi) / si_axis_max) % si_axis_max;
    int xi = (((cell_id - zi) / si_axis_max) - yi) / si_axis_max;
    int xmin, xmax, ymin, ymax, zmin, zmax;
    if (xi > 0) xmin = xi - 1; 
    else xmin = xi;
    if (xi < si_axis_x - 1) xmax = xi + 1;
    else xmax = xi;
    if (yi > 0) ymin = yi - 1; 
    else ymin = yi;
    if (yi < si_axis_y - 1) ymax = yi + 1;
    else ymax = yi;
    if (zi > 0) zmin = zi - 1; 
    else zmin = zi;
    if (zi < si_axis_z - 1) zmax = zi + 1;
    else zmax = zi;
    for (int i = xmin; i <= xmax; i++) {
        for (int j = ymin; j <= ymax; j++) {
            for (int k = zmin; k <= zmax; k++) {
                if (i != xi || j != yi || k != zi) {
                    neighbors.push_back(calculateSpatialIndex(i, j, k));
                }
            }
        }
    }
    return neighbors;
}

vector<int> getPointsWithinDistance(Eigen::RowVector3d& p, double h) {

    assert(h >= 0);
    int pid = getSpatialIndex(p);
    int search_layer = (int) ceil(h / si_dim);
    vector<int> points;
    set<int> sis;
    list<int> si_current_level;
    set<int> si_next_level;
    si_current_level.push_back(pid);
    for (int i = 0; i <= search_layer; i++) {
        while (si_current_level.size() > 0) {
            int index_current = si_current_level.front();
            sis.insert(index_current);
            auto iter_search = si_constrained_map.find(index_current);
            if (iter_search != si_constrained_map.end()) {
                vector<int> points_to_add = iter_search -> second;
                for (int point_id : points_to_add) {
                    Eigen::RowVector3d point_to_add = constrained_points.row(point_id);
                    double current_distance = getDistance(p, point_to_add);
                    if (current_distance <= h) {
                        points.push_back(point_id);
                    } 
                }
            }
            si_current_level.pop_front();
        }
        if (i < search_layer) {
            for (int gc_si : sis) {
                vector<int> gc_neighbors = getNeighborCells(gc_si);
                for (int nb : gc_neighbors) {
                    if (sis.find(nb) == sis.end() 
                      && std::find(si_current_level.begin(), si_current_level.end(), nb) == si_current_level.end()) {
                        si_next_level.insert(nb);
                    }
                }
            }
            si_current_level.insert(si_current_level.end(), si_next_level.begin(), si_next_level.end());
            si_next_level.clear();
        }
    }
    return points;

}

vector<int> getPointsWithinDistanceBrutal(Eigen::RowVector3d& p, double h) {
    vector<int> points;
    for (int i = 0; i < constrained_points.rows(); i++) {
        Eigen::RowVector3d current_point = constrained_points.row(i);
        if (getDistance(p, current_point) <= h) {
            points.push_back(i);
        }
    }
    return points;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cout << "Usage ex2_bin mesh.off" << endl;
        exit(0);
    }

    // Read points and normals
    igl::readOFF(argv[1],P,F,N);

    Viewer viewer;
    viewer.callback_key_down = callback_key_down;

    // Attach a menu plugin
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);

    menu.callback_draw_viewer_menu = [&]() {
        menu.draw_viewer_menu();
        // Add widgets to the sidebar.
        if (ImGui::CollapsingHeader("Reconstruction Options", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::InputScalar("resolution", ImGuiDataType_U32, &resolution);
            if (ImGui::Button("Reset Grid", ImVec2(-1,0))) {
                // Recreate the grid
                createGrid();
                // Switch view to show the grid
                callback_key_down(viewer, '3', 0);
            }
            // TODO: Add more parameters to tweak here...
            ImGui::InputScalar("polynomial degree", ImGuiDataType_U32, &polyDegree);
            ImGui::InputScalar("Wendland radius", ImGuiDataType_Double, &wendlandRadius);
        }

    };

    viewer.launch();
}
