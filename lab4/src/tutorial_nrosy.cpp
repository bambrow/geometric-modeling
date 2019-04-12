// Copyright (C) 2016 Daniele Panozzo <daniele.panozzo@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#include "tutorial_nrosy.h"
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/Eigenvalues>
#include <iostream>

#include "mesh_param.h"

using namespace std;
using namespace Eigen;

MatrixXd tutorial_nrosy
        (
        const MatrixXd& V,          // Vertices of the mesh
        const MatrixXi& F,          // Faces
        const MatrixXi& TT,         // Adjacency triangle-triangle
        const VectorXi& soft_id,    // Soft constraints face ids
        const MatrixXd& soft_value, // Soft constraints 3d vectors
        const int n                 // Degree of the n-rosy field
        )
{
  assert(soft_id.size() > 0); // One constraint is necessary to make the solution unique

  // This code works only for n==1, see tutorial_nrosy_complete for a generic implementation that works for n >= 1
  assert(n==1);

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
  std::vector< Triplet<std::complex<double> > > tb;

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

  // Convert the constraints into the complex polynomial coefficients and add them as soft constraints
  double lambda = 10e6;
  for (unsigned r=0; r<soft_id.size(); ++r)
  {
    int f = soft_id(r);
    Vector3d v = soft_value.row(r);
    std::complex<double> c(v.dot(T1.row(f)),v.dot(T2.row(f)));
    t.push_back(Triplet<std::complex<double> >(count,f, sqrt(lambda)));
    tb.push_back(Triplet<std::complex<double> >(count,0, c * std::complex<double>(sqrt(lambda),0)));
    ++count;
  }

  // Solve the linear system
  typedef SparseMatrix<std::complex<double>> SparseMatrixXcd;
  SparseMatrixXcd A(count,F.rows());
  A.setFromTriplets(t.begin(), t.end());
  SparseMatrixXcd b(count,1);
  b.setFromTriplets(tb.begin(), tb.end());
  SimplicialLDLT< SparseMatrixXcd > solver;
  solver.compute(A.adjoint()*A);
  assert(solver.info()==Success);
  MatrixXcd u = solver.solve(A.adjoint()*MatrixXcd(b));
  assert(solver.info()==Success);

  // Debugging informations
  #ifdef DEBUG_1
  cout << "V: " << V.rows() << " * " << V.cols() << endl;
  cout << "F: " << F.rows() << " * " << F.cols() << endl;
  cout << "TT: " << TT.rows() << " * " << TT.cols() << endl;
  cout << "soft_id: " << soft_id.rows() << " * " << soft_id.cols() << endl;
  cout << "soft_value: " << soft_value.rows() << " * " << soft_value.cols() << endl;
  cout << "T1: " << T1.rows() << " * " << T1.cols() << endl;
  cout << "T2: " << T2.rows() << " * " << T2.cols() << endl;
  cout << "t: " << t.size() << endl;
  cout << "tb: " << tb.size() << endl;
  cout << "count = " << count << endl;
  cout << "A: " << A.rows() << " * " << A.cols() << endl;
  cout << "A*: " << A.adjoint().rows() << " * " << A.adjoint().cols() << endl;
  cout << "b: " << b.rows() << " * " << b.cols() << endl;
  cout << "u: " << u.rows() << " * " << u.cols() << endl;
  #endif

  // Convert the interpolated polyvector into Euclidean vectors
  MatrixXd R(F.rows(),3);
  for (int f=0; f<F.rows(); ++f)
    R.row(f) = T1.row(f) * u(f).real() + T2.row(f) * u(f).imag();
  
  #ifdef DEBUG_4
  cout << "final locations and values: ";
  for (int i = 0; i < soft_id.size(); i++) {
      cout << soft_id(i) << "," << u(soft_id(i)) << " ";
  }
  cout << endl;
  cout << "final locations and coordinates: ";
  for (int i = 0; i < soft_id.size(); i++) {
      cout << soft_id(i) << "," << R.row(soft_id(i)) << " ";
  }
  cout << endl;
  cout << "original locations and coordinates: ";
  for (int i = 0; i < soft_id.size(); i++) {
      cout << soft_id(i) << "," << soft_value.row(i) << " ";
  }
  cout << endl;
  #endif
  
  return R;
}