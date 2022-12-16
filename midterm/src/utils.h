#ifndef _UTILS_H_
#define _UTILS_H_

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <iomanip>
using namespace std;
using namespace Eigen;

MatrixXd _get_extended_X(const MatrixXd& X);
double _calc_loss(const VectorXd& Y, const VectorXd& Y_hat);
string to_string_precision(const double in_num, const int pre);
#endif