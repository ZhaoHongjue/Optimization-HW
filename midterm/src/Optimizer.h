#ifndef _OPTIMIZER_H_
#define _OPTIMIZER_H_

#include <iostream>
#include <eigen3/Eigen/Dense>
#include "LinearModel.h"
#include "Data.h"
using namespace std;
using namespace Eigen;

class Optimizer {
protected:
    LinearModel lin;
    double lr;
    double w_decay;
public:
    Optimizer(LinearModel lin, double lr, double w_decay): lin(lin), lr(lr), w_decay(w_decay) {};
    virtual ~Optimizer() {};
    virtual void optimize(double eps, int mode) = 0;
    double get_step_size(const VectorXd& d);
};

class GradDescent : public Optimizer {
public:
    GradDescent(LinearModel lin, double lr, double w_decay): Optimizer(lin, lr, w_decay) {};
    void optimize(double eps, int mode);
};

class ConjGrad : public Optimizer {
public:
    ConjGrad(LinearModel lin, double lr, double w_decay): Optimizer(lin, lr, w_decay) {};
    void optimize(double eps, int mode);
    
    double Dai_Yuan(const VectorXd& grad1, const VectorXd& grad2, const VectorXd& p);
    double FR(const VectorXd& grad1, const VectorXd& grad2);
    double PR(const VectorXd& grad1, const VectorXd& grad2);
};

class quasiNetwon : public Optimizer {
public:
    quasiNetwon(LinearModel lin, double lr, double w_decay): Optimizer(lin, lr, w_decay) {};
    void optimize(double eps, int mode);

    MatrixXd Rank1(const MatrixXd& H, const VectorXd& delta, const VectorXd& gamma);
    MatrixXd DFP(const MatrixXd& H, const VectorXd& delta, const VectorXd& gamma);
    MatrixXd BFGS(const MatrixXd& H, const VectorXd& delta, const VectorXd& gamma);
};

#endif