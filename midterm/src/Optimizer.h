#ifndef _OPTIMIZER_H_
#define _OPTIMIZER_H_

#include <iostream>
#include <eigen3/Eigen/Dense>
#include "LinearModel.h"
#include "Data.h"
using namespace std;
using namespace Eigen;

class GradDescent {
private:
    double lr;
    double lambda;
public:
    GradDescent(double lr, double lambda): lr(lr), lambda(lambda){};
    ~GradDescent() {};
    void optimize(LinearModel& lin, Data data, double eps);
};

class ConjGrad {
private:
    double lr;
    double lambda;
public:
    ConjGrad(double lr, double lambda): lr(lr), lambda(lambda){};
    ~ConjGrad() {};
    void optimize(LinearModel& lin, Data data, double eps, int mode);
    
    double Dai_Yuan(const VectorXd& grad1, const VectorXd& grad2, const VectorXd& p);
    double FR(const VectorXd& grad1, const VectorXd& grad2);
    double PR(const VectorXd& grad1, const VectorXd& grad2);
};

class quasiNetwon {
private:
    double lr;
    double lambda;
public:
    quasiNetwon(double lr, double lambda): lr(lr), lambda(lambda){};
    ~quasiNetwon() {};
    void optimize(LinearModel& lin, Data data, double eps, int mode);

    MatrixXd Rank1(const MatrixXd& H, const VectorXd& delta, const VectorXd& gamma);
    MatrixXd DFP(const MatrixXd& H, const VectorXd& delta, const VectorXd& gamma);
    MatrixXd BFGS(const MatrixXd& H, const VectorXd& delta, const VectorXd& gamma);
};

#endif