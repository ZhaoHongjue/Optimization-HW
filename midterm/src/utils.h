#ifndef _UTILS_H_
#define _UTILS_H_

#include <iostream>
#include <eigen3/Eigen/Dense>
using namespace std;
using namespace Eigen;

MatrixXd _get_extended_X(const MatrixXd& X);
double _calc_loss(const VectorXd& Y, const VectorXd& Y_hat);
VectorXd _get_gradient(const MatrixXd& X, const VectorXd& Y, const VectorXd& weights, double lambda);

#endif