#include <iostream>
#include <eigen3/Eigen/Dense>
#include <iomanip>
#include "utils.h"
using namespace std;
using namespace Eigen;

MatrixXd _get_extended_X(const MatrixXd& X) {
    int X_rows = X.rows();
    int X_cols = X.cols();
    MatrixXd X_extended = MatrixXd::Constant(X_rows, X_cols + 1, 1);
    X_extended.block(0, 1, X_rows, X_cols) = X;
    return X_extended;
}

double _calc_loss(const VectorXd& Y, const VectorXd& Y_hat) {
    VectorXd err = Y - Y_hat;
    return (err.transpose() * err)(0, 0) / err.size();
}

string to_string_precision(const double in_num, const int pre) {
	ostringstream out_str;
	out_str << setiosflags(ios::fixed) << std::setprecision(pre) << in_num;    
	return out_str.str();
}
