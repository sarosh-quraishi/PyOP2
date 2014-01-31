static inline void lhs_cell_integral_0_otherwise (double A[3][3] , double **vertex_coordinates , double **w0 )
{
  double J[4];
  J[0] = vertex_coordinates[1][0] - vertex_coordinates[0][0]; J[1] = vertex_coordinates[2][0] - vertex_coordinates[0][0]; J[2] = vertex_coordinates[4][0] - vertex_coordinates[3][0]; J[3] = vertex_coordinates[5][0] - vertex_coordinates[3][0];;
  double K[4];
  double detJ;
  detJ = J[0]*J[3] - J[1]*J[2]; K[0] = J[3] / detJ; K[1] = -J[1] / detJ; K[2] = -J[2] / detJ; K[3] = J[0] / detJ;;
  const double det = fabs(detJ);
  static const double W3[3] = {0.166666666666667, 0.166666666666667, 0.166666666666667};
  static const double FE0_D10[3][3] = {{-1.0, 1.0, 0.0},
  {-1.0, 1.0, 0.0},
  {-1.0, 1.0, 0.0}};
  static const double FE1_C0[3][6] = {{0.666666666666667, 0.166666666666667, 0.166666666666667, 0.0, 0.0, 0.0},
  {0.166666666666667, 0.166666666666667, 0.666666666666667, 0.0, 0.0, 0.0},
  {0.166666666666667, 0.666666666666667, 0.166666666666667, 0.0, 0.0, 0.0}};
  static const double FE0_D01[3][3] = {{-1.0, 0.0, 1.0},
  {-1.0, 0.0, 1.0},
  {-1.0, 0.0, 1.0}};
  static const double FE0[3][3] = {{0.666666666666667, 0.166666666666667, 0.166666666666667},
  {0.166666666666667, 0.166666666666667, 0.666666666666667},
  {0.166666666666667, 0.666666666666667, 0.166666666666667}};
  static const double FE1_C1[3][6] = {{0.0, 0.0, 0.0, 0.666666666666667, 0.166666666666667, 0.166666666666667},
  {0.0, 0.0, 0.0, 0.166666666666667, 0.166666666666667, 0.666666666666667},
  {0.0, 0.0, 0.0, 0.166666666666667, 0.666666666666667, 0.166666666666667}};
  for (int ip = 0; ip<3; ip++)
  {
    double F0 = 0.0;
    double F1 = 0.0;
    for (int r = 0; r<6; r++)
    {
      F0 += (w0[r][0]*FE1_C0[ip][r]);
      F1 += (w0[r][0]*FE1_C1[ip][r]);
    }
#pragma pyop2 itspace
    for (int j = 0; j<3; j++)
    {
#pragma pyop2 itspace
      for (int k = 0; k<3; k++)
      {
        A[j][k] += (((FE0[ip][k]*FE0[ip][j])+(((((((K[1]*FE0_D10[ip][k])+(K[3]*FE0_D01[ip][k]))*((K[1]*FE0_D10[ip][j])+(K[3]*FE0_D01[ip][j])))+(((K[0]*FE0_D10[ip][k])+(K[2]*FE0_D01[ip][k]))*((K[0]*FE0_D10[ip][j])+(K[2]*FE0_D01[ip][j]))))*0.1)+(((((K[1]*FE0_D10[ip][j])+(K[3]*FE0_D01[ip][j]))*F1)+(((K[0]*FE0_D10[ip][j])+(K[2]*FE0_D01[ip][j]))*F0))*FE0[ip][k]*-1.0))*5e-05))*det*W3[ip]);
      }
    }
  }
}