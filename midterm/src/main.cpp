#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <eigen3/Eigen/Dense>

#include "Data.h"
#include "LinearModel.h"
#include "Optimizer.h"
#include "Recorder.h"
using namespace std;
using namespace Eigen;

int main()
{   
    cout << "<===================Start===================>" << endl;
    double lambda = 10;
    double lr = 1;
    double eps = 1e-9;
    double split_rate = 0.8;
    
    Data data(split_rate);
    // data.read_data("./data/housing.txt");
    data.read_data("./data/bodyfat_scale.txt");
    LinearModel lin(data);
    // cout << lin.hess(lambda) << endl;
    
    GradDescent GD(lin, lr, lambda);
    Recorder r1("bodyfat.csv");
    GD.optimize(r1, eps, 0);

    // quasiNetwon qN(lin, lr, lambda);
    // qN.optimize(eps, 2);

    // ConjGrad CG(lin, lr, lambda);
    // CG.optimize(eps, 0);

    // double h = GD.Armijo(-1*lin.gradient(lambda), lr, 0.5, 1);
    // cout << h << endl;
    // double lambda = 0.0;
    // double lr = 0.0001;
    // 
    // VectorXd w1 = lin.get_weights();

    // GradDescent GD(lin, lr, lambda);
    // GD.optimize(data, 1e-3);

    

    
    
    // 
    // cout << "Loss with lambda=" << lambda << ": " << _calc_loss(data.get_Y(), Y_hat) << endl;
    // cout << "Gradient:" << endl;
    cout << "<====================End====================>" << endl;
}