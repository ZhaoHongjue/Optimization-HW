#ifndef _LINEAR_MODEL_H
#define _LINEAR_MODEL_H

#include <iostream>
#include <eigen3/Eigen/Dense>
#include "Data.h"
#include "utils.h"
using namespace std;
using namespace Eigen;

class LinearModel {
private:
    VectorXd weights;
    Data data;
public:
    LinearModel(const Data& data);
    LinearModel(const Data& data, const VectorXd& weights): data(data), weights(weights) {};
    ~LinearModel() {};

    VectorXd forward(const MatrixXd& X);
    VectorXd forward() {return forward(get_X());};
    VectorXd gradient(double lambda);
    MatrixXd hess(double lambda);
    MatrixXd get_X() {return this->data.get_X();};
    VectorXd get_Y() {return this->data.get_Y();};
    VectorXd get_weights() {return this->weights;};
    void set_weights(VectorXd weights) {this->weights = weights;};
    void set_ideal_weights(double w_decay) {
        this->weights = data.calc_ana_solution(w_decay);
    };
};

#endif