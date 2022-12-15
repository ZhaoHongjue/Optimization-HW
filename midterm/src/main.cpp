#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <eigen3/Eigen/Dense>

#include "Data.h"
// #include "Data.h"
// #include "LinearModel.h"
#include "Optimizer.h"
// #include "utils.h"
using namespace std;
using namespace Eigen;

int main()
{   
    cout << "<===================Start===================>" << endl;
    double lambda = 0.01;
	Data data;
    data.read_data("./data/housing.txt");
    auto ideal_weights = data.calc_ana_solution(lambda);
    cout << ideal_weights << endl;
    cout << _get_gradient(data.get_X(), data.get_Y(), ideal_weights, lambda).norm() << endl;
    cout << "--------------------------------------------" << endl;

    LinearModel lin(data.get_feature_num());

    GradDescent GD(0.00001, lambda);
    GD.optimize(lin, data, 1e-9);

    // ConjGrad CG(0.0000001, lambda);
    // CG.optimize(lin, data, 1e-9, 2);

    // quasiNetwon qN(0.000001, lambda);
    // qN.optimize(lin, data, 1e-3, 1);
    
    // 
    // cout << "Loss with lambda=" << lambda << ": " << _calc_loss(data.get_Y(), Y_hat) << endl;
    // cout << "Gradient:" << endl;
    cout << "<====================End====================>" << endl;
}