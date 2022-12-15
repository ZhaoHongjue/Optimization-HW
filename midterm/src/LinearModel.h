#ifndef _LINEAR_MODEL_H
#define _LINEAR_MODEL_H

#include <iostream>
#include <eigen3/Eigen/Dense>
#include "utils.h"
using namespace std;
using namespace Eigen;

class LinearModel {
private:
    VectorXd weights;
public:
    LinearModel(int num_feature);
    LinearModel(const VectorXd& ideal_weights);
    ~LinearModel() {};

    VectorXd forward(const MatrixXd& X);
    VectorXd get_weights() {return this->weights;};
    void set_weights(VectorXd weights) {this->weights = weights;};
};

#endif