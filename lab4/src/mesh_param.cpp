
#include "mesh_param.h"

#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <igl/grad.h>
#include <igl/slice.h>
#include <igl/cat.h>
#include <igl/doublearea.h>
#include <igl/harmonic.h>
#include <igl/lscm.h>
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <set>

using namespace std;
using namespace Eigen;

MatrixXd interpolate_field
(
    const MatrixXd& V,          // Vertices of the mesh
    const MatrixXi& F,          // Faces
    const MatrixXi& TT,         // Adjacency triangle-triangle
    const VectorXi& b,          // Constrained faces id
    const MatrixXd& bc          // Cosntrained faces representative vector
) 
{
    // Lots of the codes below are based on the codes in tutorial_nrosy.cpp
    // However, there are a lot of modifications

    // Starter code, same as template
    assert(b.size() > 0); // One constraint is necessary to make the solution unique
    Matrix<double,Eigen::Dynamic,3> T1(F.rows(),3), T2(F.rows(),3);
    // Compute the local reference systems for each face
    for (unsigned i=0;i<F.rows();++i)
    {
        Vector3d e1 =  V.row(F(i, 1)) - V.row(F(i, 0));
        Vector3d e2 =  V.row(F(i, 2)) - V.row(F(i, 0));
        T1.row(i) = e1.normalized();
        T2.row(i) = T1.row(i).cross(T1.row(i).cross(e2)).normalized();
    }

    // Build the sparse matrix, with an energy term for each edge
    std::vector< Triplet<std::complex<double> > > t;
    // The b vector is all zeroes so we do not need it here
    // std::vector< Triplet<std::complex<double> > > tb;

    // Handle the free part of matrix A; same as template
    unsigned count = 0;
    for (unsigned f=0;f<F.rows();++f)
    {
        for (unsigned ei=0;ei<F.cols();++ei)
        {
            // Look up the opposite face
            int g = TT(f,ei);
            // If it is a boundary edge, it does not contribute to the energy
            if (g == -1) continue;
            // Avoid to count every edge twice
            if (f > g) continue;
            // Compute the complex representation of the common edge
            Vector3d e  = (V.row(F(f,(ei+1)%3)) - V.row(F(f,ei)));
            Vector2d vef = Vector2d(e.dot(T1.row(f)),e.dot(T2.row(f))).normalized();
            std::complex<double> ef(vef(0),vef(1));
            Vector2d veg = Vector2d(e.dot(T1.row(g)),e.dot(T2.row(g))).normalized();
            std::complex<double> eg(veg(0),veg(1));
            // Add the term conj(f)^n*ui - conj(g)^n*uj to the energy matrix
            t.push_back(Triplet<std::complex<double> >(count,f,    std::conj(ef)));
            t.push_back(Triplet<std::complex<double> >(count,g,-1.*std::conj(eg)));
            ++count;
        }
    }

    // Since tb is not used, comment out the codes related to tb
    // double lambda = 10e6;
    double lambda = 0;
    for (unsigned r=0; r<b.size(); ++r)
    {
        int f = b(r);
        // Vector3d v = bc.row(r);
        // std::complex<double> c(v.dot(T1.row(f)),v.dot(T2.row(f)));
        t.push_back(Triplet<std::complex<double> >(count,f, sqrt(lambda)));
        // tb.push_back(Triplet<std::complex<double> >(count,0, c * std::complex<double>(sqrt(lambda),0)));
        ++count;
    }

    const unsigned fr = F.rows();
    const unsigned bs = b.size();

    // Solve the linear system
    // Here the solving is based on the equation
    // Aff * u = - Afc * xc
    // (#f is the number of total faces)
    // (#b is number of constrained faces)
    // (so #f-#b is the number of free faces)
    // Where Aff is (#f-#b)*(#f-#b) matrix
    //       u is our answer, (#f-#b)*#1 vector
    //       Afc is (#f-#b)*#b matrix 
    //       xc is #b*#1 vector
    // Therefore, the right part is (#f-#b)*#1 vector
    // We need A to be a matrix of #count*#f
    // So that Q=A.adjoint()*A is #f*#f
    // Aff is the (#f-#b)*(#f-#b) part of Q
    // Afc is the (#f-#b)*#b part of Q
    typedef SparseMatrix<std::complex<double>> SparseMatrixXcd;
    // SparseMatrixXcd A(count,F.rows());
    SparseMatrixXcd A(count,fr); // #count*#f
    A.setFromTriplets(t.begin(), t.end());
    VectorXcd xc(bs,1); // #b*#1
    for (unsigned r=0; r<b.size(); ++r) {
        int f = b(r);
        Vector3d v = bc.row(r);
        std::complex<double> c(v.dot(T1.row(f)),v.dot(T2.row(f)));
        xc(r) = c;
    }

    #ifdef DEBUG_3
    for (unsigned r=0; r<b.size(); ++r) {
        int f = b(r);
        cout << "representation for vector on face " << f << ": ";
        cout << bc.row(r) << ", " << xc(r) << ", ";
        cout << T1.row(f) * xc(r).real() + T2.row(f) * xc(r).imag() << endl;
    }
    #endif

    SparseMatrixXcd Q = A.adjoint()*A; // #f*#f

    // Get a set of constrained faces
    set<int> constrained_set;
    for (unsigned r=0; r<b.size(); ++r) {
        int f = b(r);
        constrained_set.insert(f);
    }

    // Get the vector representing free faces
    VectorXi vf(fr-bs);
    unsigned findex = 0;
    for (unsigned f=0;f<F.rows();++f) {
        if (constrained_set.find(f) == constrained_set.end()) {
            vf(findex) = f;
            findex++;
        }
    }

    #ifdef DEBUG_5
    cout << "constrained faces: " << endl;
    cout << b << endl;
    cout << xc << endl;
    cout << "--------------" << endl;
    cout << "free faces: " << endl;
    cout << vf << endl;
    #endif

    // Construct Aff
    SparseMatrixXcd Aff(fr-bs, fr-bs); // (#f-#b)*(#f-#b)
    igl::slice(Q, vf, vf, Aff); 

    // Construct Afc
    SparseMatrixXcd Afc(fr-bs, bs); // (#f-#b)*#b
    igl::slice(Q, vf, b, Afc);

    MatrixXcd mb = (Afc * xc) * -1; // (#f-#b)*#1
    // SparseMatrixXcd b(count,1);
    // b.setFromTriplets(tb.begin(), tb.end());
    SimplicialLDLT< SparseMatrixXcd > solver;
    solver.compute(Aff);
    // solver.compute(A.adjoint()*A);
    assert(solver.info()==Success);
    MatrixXcd u = solver.solve(mb);
    // MatrixXcd u = solver.solve(A.adjoint()*MatrixXcd(b));
    assert(solver.info()==Success);

    // Finally, rearrange u and store values in ua
    VectorXcd ua(fr);
    for (unsigned ui = 0; ui < u.size(); ui++) {
        ua(vf(ui)) = u(ui);
    }
    for (unsigned r=0; r<b.size(); ++r) {
        ua(b(r)) = xc(r);
    }

    #ifdef DEBUG_3
    cout << "final insertion locations and values: ";
    for (int i = 0; i < bs; i++) {
        cout << b(i) << "," << xc(i) << " ";
    }
    cout << endl;
    #endif

    // Debugging informations
    #ifdef DEBUG_2
    cout << "--------------" << endl;
    cout << "V: " << V.rows() << " * " << V.cols() << endl;
    cout << "F: " << F.rows() << " * " << F.cols() << endl;
    cout << "TT: " << TT.rows() << " * " << TT.cols() << endl;
    cout << "b: " << b.rows() << " * " << b.cols() << endl;
    cout << "vf: " << vf.rows() << " * " << vf.cols() << endl;
    cout << "bc: " << bc.rows() << " * " << bc.cols() << endl;
    cout << "T1: " << T1.rows() << " * " << T1.cols() << endl;
    cout << "T2: " << T2.rows() << " * " << T2.cols() << endl;
    cout << "t: " << t.size() << endl;
    cout << "count = " << count << endl;
    cout << "A: " << A.rows() << " * " << A.cols() << endl;
    cout << "Q: " << Q.rows() << " * " << Q.cols() << endl;
    cout << "xc: " << xc.rows() << " * " << xc.cols() << endl;
    cout << "Aff: " << Aff.rows() << " * " << Aff.cols() << endl;
    cout << "Afc: " << Afc.rows() << " * " << Afc.cols() << endl;
    cout << "mb: " << mb.rows() << " * " << mb.cols() << endl;
    cout << "u: " << u.rows() << " * " << u.cols() << endl;
    cout << "ua: " << ua.rows() << " * " << ua.cols() << endl;
    cout << "--------------" << endl;
    cout << "Q nonzeros: " << Q.nonZeros() << endl;
    cout << "Aff nonzeros: " << Aff.nonZeros() << endl;
    cout << "Afc nonzeros: " << Afc.nonZeros() << endl;
    cout << "--------------" << endl;
    #endif

    // Convert the interpolated polyvector into Euclidean vectors
    MatrixXd R(F.rows(),3);
    for (int f=0; f<ua.size(); ++f) {
        R.row(f) = T1.row(f) * ua(f).real() + T2.row(f) * ua(f).imag();
    }

    #ifdef DEBUG_3
    cout << "final insertion locations and coordinates: ";
    for (int i = 0; i < bs; i++) {
        cout << b(i) << "," << R.row(b(i)) << " ";
    }
    cout << endl;
    #endif

    #ifdef DEBUG_3
    cout << "original locations and coordinates: ";
    for (int i = 0; i < bs; i++) {
        cout << b(i) << "," << bc.row(i) << " ";
    }
    cout << endl;
    #endif
    
    return R;
}

MatrixXd get_scalar_field
(
    const MatrixXd& V,             // Vertices of the mesh
    const MatrixXi& F,             // Faces
    const MatrixXd& R,             // Vector field
    const SparseMatrix<double>& G  // Gradient operator 
) 
{
    // Construct matrix A (areas)
    VectorXd A1(F.rows());
    igl::doublearea(V,F,A1);
    SparseMatrix<double> A(F.rows()*3, F.rows()*3); // #3f*#3f
    for (unsigned f = 0; f < F.rows(); ++f) {
        A.insert(f,f) = A1(f);
        A.insert(f+F.rows(),f+F.rows()) = A1(f);
        A.insert(f+F.rows()*2,f+F.rows()*2) = A1(f);
    }

    // Construct matrix u (from R)
    VectorXd u(F.rows()*3); // #3f*#1
    for (unsigned f = 0; f < F.rows(); ++f) {
        u(f) = R(f,0);
        u(f+F.rows()) = R(f,1);
        u(f+2*F.rows()) = R(f,2);
    }

    // Construct K
    SparseMatrix<double> K = G.transpose() * A * G; // #v*#v
    // Construct b
    VectorXd b = G.transpose() * A * u; // #v*#1

    // Row/col removal
    SparseMatrix<double> Kff = K.bottomRightCorner(K.rows()-1, K.cols()-1);
    // SparseMatrix<double> Kfc = K.bottomLeftCorner(K.rows()-1, 1);
    // VectorXd sc(1);
    // sc << 0;
    // VectorXd bt = b.bottomRows(b.rows()-1) - Kfc * sc;
    VectorXd bt = b.bottomRows(b.rows()-1);

    #ifdef DEBUG_7
    cout << "--------------" << endl;
    cout << "V: " << V.rows() << " * " << V.cols() << endl;
    cout << "F: " << F.rows() << " * " << F.cols() << endl;
    cout << "R: " << R.rows() << " * " << R.cols() << endl;
    cout << "G: " << G.rows() << " * " << G.cols() << endl;
    cout << "A: " << A.rows() << " * " << A.cols() << endl;
    cout << "u: " << u.rows() << " * " << u.cols() << endl;
    cout << "K: " << K.rows() << " * " << K.cols() << endl;
    cout << "b: " << b.rows() << " * " << b.cols() << endl;
    cout << "--------------" << endl;
    cout << "G nonzeros: " << G.nonZeros() << endl;
    cout << "A nonzeros: " << A.nonZeros() << endl;
    cout << "K nonzeros: " << K.nonZeros() << endl;
    #endif

    // Solve the system
    SimplicialLDLT<SparseMatrix<double>> solver;
    // solver.compute(K);
    // assert(solver.info() == Success);
    // MatrixXd s = solver.solve(b);
    // assert(solver.info() == Success);
    solver.compute(Kff);
    assert(solver.info() == Success);
    MatrixXd sf = solver.solve(bt);
    assert(solver.info() == Success);
    VectorXd s(K.rows());
    s(0) = 0;
    for (unsigned i = 1; i < K.rows(); i++) {
        s(i) = sf(i-1);
    }

    // Debugging informations
    #ifdef DEBUG_7
    cout << "s: " << s.rows() << " * " << s.cols() << endl;
    cout << "--------------" << endl;
    #endif

    return s;
}

MatrixXd harmonic_param
(
    const MatrixXd& V,             // Vertices of the mesh
    const MatrixXi& F              // Faces
) 
{
    // Find the open boundary
    VectorXi bnd;
    igl::boundary_loop(F, bnd);
    // Map the boundary to a circle
    MatrixXd bnd_uv;
    igl::map_vertices_to_circle(V, bnd, bnd_uv);
    // Harmonic parameterizations
    MatrixXd V_uv;
    igl::harmonic(V,F,bnd,bnd_uv,1,V_uv);
    V_uv *= 8;

    #ifdef DEBUG_9
    cout << "--------------" << endl;
    cout << "V: " << V.rows() << " * " << V.cols() << endl;
    cout << "F: " << F.rows() << " * " << F.cols() << endl;
    cout << "bnd: " << bnd.rows() << " * " << bnd.cols() << endl;
    cout << "bnd_uv: " << bnd_uv.rows() << " * " << bnd_uv.cols() << endl;
    cout << "V_uv: " << V_uv.rows() << " * " << V_uv.cols() << endl;
    cout << "--------------" << endl;
    #endif

    return V_uv;
}

MatrixXd lscm_param
(
    const MatrixXd& V,             // Vertices of the mesh
    const MatrixXi& F              // Faces
) 
{
    // Fix two points on the boundary
    VectorXi bnd, b(2,1);
    igl::boundary_loop(F,bnd);
    b(0) = bnd(0);
    b(1) = bnd(round(bnd.size()/2));
    MatrixXd bc(2,2);
    bc << 0,0,1,0;
    // LSCM parametrizations
    MatrixXd V_uv;
    igl::lscm(V,F,b,bc,V_uv);
    V_uv *= 8;

    #ifdef DEBUG_9
    cout << "--------------" << endl;
    cout << "V: " << V.rows() << " * " << V.cols() << endl;
    cout << "F: " << F.rows() << " * " << F.cols() << endl;
    cout << "bnd: " << bnd.rows() << " * " << bnd.cols() << endl;
    cout << "V_uv: " << V_uv.rows() << " * " << V_uv.cols() << endl;
    cout << "--------------" << endl;
    #endif

    return V_uv;
}

