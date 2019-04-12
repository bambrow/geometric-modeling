// Copyright (C) 2016 Daniele Panozzo <daniele.panozzo@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#include <igl/avg_edge_length.h>
#include <igl/barycenter.h>
#include <igl/local_basis.h>
#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/grad.h>
#include <igl/jet.h>

#include "tutorial_nrosy.h"
#include "mesh_param.h"
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

// Mesh
Eigen::MatrixXd V;
Eigen::MatrixXi F;

// Triangle-triangle adjacency
Eigen::MatrixXi TT;
Eigen::MatrixXi TTi;

// Constrained faces id
Eigen::VectorXi b;

// Cosntrained faces representative vector
Eigen::MatrixXd bc;

// Currently selected face
int selected;

// Degree of the N-RoSy field
int N;

// Local basis
Eigen::MatrixXd B1, B2, B3;

// Additional global variables
unsigned char current_key; // stores the value of current key
Eigen::MatrixXd RM; // stores the computed vector field
Eigen::MatrixXd UV; // stores the computed uv field

// Texture image (grayscale)
Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> texture_I;
void line_texture() 
{
  int size = 128;              // Texture size
  int w    = 7;                // Line width
  int pos  = size / 2 - w / 2; // Center the line
  texture_I.setConstant(size, size, 255);
  texture_I.block(0, pos, size, w).setZero();
  texture_I.block(pos, 0, w, size).setZero();

}
// Converts a representative vector per face in the full set of vectors that describe
// an N-RoSy field
void representative_to_nrosy(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const Eigen::MatrixXd& R,
  const int N,
  Eigen::MatrixXd& Y)
{
  using namespace Eigen;
  using namespace std;

  Y.resize(F.rows()*N,3);
  for (unsigned i=0;i<F.rows();++i)
  {
    double x = R.row(i) * B1.row(i).transpose();
    double y = R.row(i) * B2.row(i).transpose();
    double angle = atan2(y,x);

    for (unsigned j=0; j<N;++j)
    {
      double anglej = angle + 2*M_PI*double(j)/double(N);
      double xj = cos(anglej);
      double yj = sin(anglej);
      Y.row(i*N+j) = xj * B1.row(i) + yj * B2.row(i);
      Y.row(i*N+j) = Y.row(i*N+j) * R.row(i).norm();
    }

  }
}

// Plots the mesh with an N-RoSy field
// The constrained faces (b) are colored in red.
void plot_mesh_nrosy(
  igl::opengl::glfw::Viewer& viewer,
  Eigen::MatrixXd& V,
  Eigen::MatrixXi& F,
  int N,
  Eigen::MatrixXd& PD1,
  Eigen::VectorXi& b)
{
  using namespace Eigen;
  using namespace std;
  // Clear the mesh
  viewer.data().clear();
  viewer.data().set_mesh(V,F);
  viewer.data().set_texture(texture_I, texture_I, texture_I);

  // Expand the representative vectors in the full vector set and plot them as lines
  double avg = igl::avg_edge_length(V, F);
  MatrixXd Y;
  representative_to_nrosy(V, F, PD1, N, Y);

  MatrixXd B;
  igl::barycenter(V,F,B);

  MatrixXd Be(B.rows()*N,3);
  for(unsigned i=0; i<B.rows();++i)
    for(unsigned j=0; j<N; ++j)
      Be.row(i*N+j) = B.row(i);

  viewer.data().add_edges(Be,Be+Y*(avg/2),RowVector3d(0,0,1));

  // Highlight in red the constrained faces
  MatrixXd C = MatrixXd::Constant(F.rows(),3,1);
  for (unsigned i=0; i<b.size();++i)
    C.row(b(i)) << 1, 0, 0;
  viewer.data().set_colors(C);
}

void write_matrix_to_file(Eigen::MatrixXd matrix, std::string filename) {
  std::ofstream fout(filename);
  if (fout.is_open()) {
    fout << matrix;
  }
  fout.close();
}

void write_vector_to_file(Eigen::VectorXi vector, std::string filename) {
  std::ofstream fout(filename);
  if (fout.is_open()) {
    fout << vector;
  }
  fout.close();
}

// It allows to change the degree of the field when a number is pressed
bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{
  using namespace Eigen;
  using namespace std;

  if (key == '1')
  {

    current_key = key;
    N = key - '0'; // N = 1
    RM.resize(0,3);
    RM = tutorial_nrosy(V,F,TT,b,bc,N);
    plot_mesh_nrosy(viewer,V,F,N,RM,b);
    viewer.data().show_texture = false;

    #ifdef DEBUG_A
    cout << RM.topRows(20) << endl;
    cout << "##########" << endl;
    #endif

    write_matrix_to_file(RM, "../1_vector_field.txt");

  } 
  else if (key == '2') 
  {

    N = 1;
    current_key = key;
    RM.resize(0,3);
    RM = interpolate_field(V,F,TT,b,bc);
    plot_mesh_nrosy(viewer,V,F,N,RM,b);
    viewer.data().show_texture = false;

    #ifdef DEBUG_A
    cout << RM.topRows(20) << endl;
    cout << "##########" << endl;
    #endif

    write_matrix_to_file(RM, "../2_vector_field.txt");

  } 
  else if (key == '3') 
  {

    N = 1;
    current_key = key;
    // Compute gradient operator G
    SparseMatrix<double> G; // #3f*#v
    igl::grad(V,F,G);

    // Get vector s
    MatrixXd s = get_scalar_field(V,F,RM,G); // #v*#1

    // Compute vector g
    MatrixXd g = G * s; // #3f*#1

    // Split g to get back the matrix of gradient vectors
    MatrixXd gm(F.rows(), 3); // #f*#3
    for (unsigned f = 0; f < F.rows(); ++f) {
      gm(f,0) = g(f);
      gm(f,1) = g(f+F.rows());
      gm(f,2) = g(f+F.rows()*2);
    }

    // Plot gradient vectors
    plot_mesh_nrosy(viewer,V,F,N,gm,b);

    // Compute deviation
    VectorXd dev_vector = (gm - RM).rowwise().norm();

    // Plot scalar field
    MatrixXd color_map;
    igl::jet(s, true, color_map);
    // IMPORTANT NOTE: uncomment the line below, and comment the line above, to plot the deviation
    // igl::jet(dev_vector, true, color_map);
    viewer.data().set_colors(color_map);
    viewer.data().show_texture = false;

    #ifdef DEBUG_8
    cout << "--------------" << endl;
    cout << "dev_vector: " << dev_vector.rows() << " * " << dev_vector.cols() << endl;
    cout << "--------------" << endl;
    #endif

    #ifdef DEBUG_A
    cout << RM.topRows(20) << endl;
    cout << "##########" << endl;
    cout << gm.topRows(20) << endl;
    cout << "##########" << endl;
    #endif

    write_matrix_to_file(s, "../3_scalar_function.txt");

  } 
  else if (key == '4') 
  {

    current_key = key;
    UV.resize(0,2);
    // Compute UV
    UV = harmonic_param(V,F);

    MatrixXd color_map;
    igl::jet(UV.col(1), true, color_map);
    viewer.data().show_texture = false;
    viewer.data().clear();
    viewer.data().set_mesh(V, F);
    viewer.data().set_texture(texture_I, texture_I, texture_I);
    viewer.data().set_colors(color_map);
    viewer.data().set_uv(UV);
    viewer.data().show_lines = false;
    viewer.data().show_texture = true;
    // viewer.data().set_mesh(V_uv,F);

  } 
  else if (key == '5') 
  {

    current_key = key;
    UV.resize(0,2);
    // Compute UV
    UV = lscm_param(V,F);

    MatrixXd color_map;
    igl::jet(UV.col(1), true, color_map);
    viewer.data().show_texture = false;
    viewer.data().clear();
    viewer.data().set_mesh(V, F);
    viewer.data().set_texture(texture_I, texture_I, texture_I);
    viewer.data().set_colors(color_map);
    viewer.data().set_uv(UV);
    viewer.data().show_lines = false;
    viewer.data().show_texture = true;
    // viewer.data().set_mesh(V_uv,F);

  }
  else if (key == '6') 
  {

    N = 1;
    current_key = key;
    // Compute gradient operator G
    SparseMatrix<double> G; // #3f*#v
    igl::grad(V,F,G);

    // Scale down the vector
    VectorXd v = UV.col(1) / 5; 
    // Compute vector g
    MatrixXd g = G * v; // #3f*#1

    // Split g to get back the matrix of gradient vectors
    MatrixXd gm(F.rows(), 3); // #f*#3
    for (unsigned f = 0; f < F.rows(); ++f) {
      gm(f,0) = g(f);
      gm(f,1) = g(f+F.rows());
      gm(f,2) = g(f+F.rows()*2);
    }

    // Plot gradient vectors
    plot_mesh_nrosy(viewer,V,F,N,gm,b);

    // Plot scalar field
    MatrixXd color_map;
    igl::jet(UV.col(1), true, color_map);
    viewer.data().set_colors(color_map);
    viewer.data().show_lines = false;
    viewer.data().show_texture = false;

  }
  else if (key == '7') 
  {

    N = 1;
    current_key = key;
    // Compute gradient operator G
    SparseMatrix<double> G; // #3f*#v
    igl::grad(V,F,G);

    // Get vector s
    MatrixXd s = get_scalar_field(V,F,RM,G); // #v*#1
    double scale = UV.col(0).norm() / s.norm();
    // Compute uv, replacing V
    MatrixXd uv(V.rows(),2);
    for (unsigned i = 0; i < V.rows(); i++) {
      uv(i,0) = UV(i,0);
      uv(i,1) = s(i) * scale;
    }
    // uv.colwise().normalize();

    // Compute vector g
    MatrixXd g = G * s; // #3f*#1

    // Split g to get back the matrix of gradient vectors
    MatrixXd gm(F.rows(), 3); // #f*#3
    for (unsigned f = 0; f < F.rows(); ++f) {
      gm(f,0) = g(f);
      gm(f,1) = g(f+F.rows());
      gm(f,2) = g(f+F.rows()*2);
    }

    // Plot gradient vectors
    plot_mesh_nrosy(viewer,V,F,N,gm,b);

    // Plot scalar field
    MatrixXd color_map;
    igl::jet(s, true, color_map);
    viewer.data().set_colors(color_map);
    viewer.data().show_texture = false;

    // Plot texture
    viewer.data().set_uv(uv);
    viewer.data().show_lines = false;
    viewer.data().show_texture = true;
    // viewer.data().set_mesh(V_uv,F);

    #ifdef DEBUG_B
    cout << UV.topRows(20) << endl;
    cout << "##########" << endl;
    cout << uv.topRows(20) << endl;
    cout << "##########" << endl;
    #endif

  }
  else if (key == '8') {

    current_key = key;
    // Compute gradient operator G
    SparseMatrix<double> G; // #3f*#v
    igl::grad(V,F,G);

    // Get vector s
    MatrixXd s = get_scalar_field(V,F,RM,G); // #v*#1
    double scale = UV.col(0).norm() / s.norm();
    // Compute uv, replacing V
    MatrixXd uv(V.rows(),2);
    for (unsigned i = 0; i < V.rows(); i++) {
      uv(i,0) = UV(i,0);
      uv(i,1) = s(i) * scale;
    }
    // uv.colwise().normalize();
    
    VectorXi flip_pos(F.rows());
    VectorXi flip_neg(F.rows());
    unsigned fpos_index = 0, fneg_index = 0;
    for (unsigned f = 0; f < F.rows(); f++) {
      VectorXi vs = F.row(f);
      VectorXd e11 = UV.row(vs(1)) - UV.row(vs(0));
      VectorXd e12 = UV.row(vs(2)) - UV.row(vs(0));
      double n1 = e11(0) * e12(1) - e11(1) * e12(0);
      VectorXd e21 = uv.row(vs(1)) - uv.row(vs(0));
      VectorXd e22 = uv.row(vs(2)) - uv.row(vs(0));
      double n2 = e21(0) * e22(1) - e21(1) * e22(0);
      if (n1 * n2 > 0) {
        flip_pos(fpos_index) = f;
        fpos_index++;
      } else {
        flip_neg(fneg_index) = f;
        fneg_index++;
      }
    }
    flip_pos.conservativeResize(fpos_index);
    flip_neg.conservativeResize(fneg_index);
    VectorXi flipped;
    if (fpos_index > fneg_index) {
      flipped = flip_neg;
    } else {
      flipped = flip_pos;
    }
    // cout << flipped << endl;

    viewer.data().clear();
    viewer.data().set_mesh(V,F);
    viewer.data().set_texture(texture_I, texture_I, texture_I);
    MatrixXd C = MatrixXd::Constant(F.rows(),3,1);
    for (unsigned i=0; i<flipped.size();++i) {
      C.row(flipped(i)) << 1, 0, 0;
    }
    viewer.data().set_colors(C);
    viewer.data().show_lines = true;
    viewer.data().show_texture = false;

    write_vector_to_file(flipped, "../8_flipped_triangles.txt");
    
  }

  if (key == '[' || key == ']')
  {
    if (selected >= b.size() || selected < 0)
      return false;

    int i = b(selected);
    Vector3d v = bc.row(selected);

    double x = B1.row(i) * v;
    double y = B2.row(i) * v;
    double norm = sqrt(x*x+y*y);
    double angle = atan2(y,x);

    angle += key == '[' ? -M_PI/16 : M_PI/16;

    double xj = cos(angle)*norm;
    double yj = sin(angle)*norm;

    bc.row(selected) = xj * B1.row(i) + yj * B2.row(i);

    key_down(viewer, current_key, 0);  
  }

  if (key == 'Q' || key == 'W')
  {
    if (selected >= b.size() || selected < 0)
      return false;

    bc.row(selected) =  bc.row(selected) * (key == 'Q' ? 3./2. : 2./3.);

    key_down(viewer, current_key, 0);
  }

  if (key == 'E')
  {
    if (selected >= b.size() || selected < 0)
      return false;

    b(selected) = b(b.rows()-1);
    b.conservativeResize(b.size()-1);
    bc.row(selected) = bc.row(bc.rows()-1);
    bc.conservativeResize(b.size(),bc.cols());
    
    key_down(viewer, current_key, 0);
  }

  return false;
}

bool mouse_down(igl::opengl::glfw::Viewer& viewer, int, int)
{
  int fid_ray;
  Eigen::Vector3f bary;
  // Cast a ray in the view direction starting from the mouse position
  double x = viewer.current_mouse_x;
  double y = viewer.core.viewport(3) - viewer.current_mouse_y;
  if(igl::unproject_onto_mesh(Eigen::Vector2f(x,y), viewer.core.view,
    viewer.core.proj, viewer.core.viewport, V, F, fid_ray, bary))
  {
    bool found = false;
    for (int i=0;i<b.size();++i)
    {
      if (b(i) == fid_ray)
      {
        found = true;
        selected = i;
      }
    }

    if (!found)
    {
      b.conservativeResize(b.size()+1);
      b(b.size()-1) = fid_ray;
      bc.conservativeResize(bc.rows()+1,bc.cols());
      bc.row(bc.rows()-1) << 1, 1, 1;
      selected = bc.rows()-1;

      key_down(viewer, current_key, 0);
    }

    return true;
  }
  return false;
};


int main(int argc, char *argv[])
{
  using namespace std;
  using namespace Eigen;

  string file_name;
  // Load a mesh in OBJ format
  if (argc <= 1) {
    file_name = "../bumpy.off";
  } else {
    file_name = argv[1];
  }
  igl::readOFF(file_name, V, F);
  line_texture();
  // Triangle-triangle adjacency
  igl::triangle_triangle_adjacency(F,TT,TTi);

  // Compute the local_basis
  igl::local_basis(V,F,B1,B2,B3);

  if (argc <= 2) {
    // Simple constraints
    b.resize(2);
    b(0) = 0;
    b(1) = F.rows()-1;
    bc.resize(2,3);
    bc << 1,1,1,0,1,1;
  } else {
    // Read constraints from file
    string constraints_name = argv[2];
    ifstream fin(constraints_name);
    string fline;
    unsigned findex = 0;
    // Assign adequate spaces first
    b.resize(F.rows());
    bc.resize(F.rows(), 3);
    while (getline(fin, fline)) {
      istringstream fiss(fline);
      int face_index;
      double face_x, face_y, face_z;
      fiss >> face_index;
      fiss >> face_x;
      fiss >> face_y;
      fiss >> face_z;
      b(findex) = face_index;
      bc(findex, 0) = face_x;
      bc(findex, 1) = face_y;
      bc(findex, 2) = face_z;
      findex++;
    }
    fin.close();
    // Resize back
    b.conservativeResize(findex);
    bc.conservativeResize(findex, 3);
    #ifdef DEBUG_6
    cout << b << endl;
    cout << bc << endl;
    #endif
  } 

  selected = 0;

  igl::opengl::glfw::Viewer viewer;

  // Interpolate the field and plot
  key_down(viewer, '1', 0);

  // Plot the mesh
  viewer.data().set_mesh(V, F);
  viewer.data().set_texture(texture_I, texture_I, texture_I);

  // Register the callbacks
  viewer.callback_key_down = &key_down;
  viewer.callback_mouse_down = &mouse_down;
  
  // Disable wireframe
  viewer.data().show_lines = false;

  // Launch the viewer
  viewer.launch();
}
