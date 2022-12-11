#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "Dataset.h"
#include "LinearModel.h"
using namespace std;

int main()
{
    double lambda = 1.0;
	Dataset dataset;
    dataset.read_data("./data/bodyfat.txt");
    auto ideal_weights = dataset.calc_ana_ridge_solution(lambda);
    // cout << ideal_weights << endl;
    
    LinearModel lin(dataset.get_feature_num());
    lin.set_weights(ideal_weights);
    auto Y_hat = lin.forward(dataset.get_X());
    cout << "Loss with lambda = " << lambda << ": " << lin.calc_loss(dataset.get_Y(), Y_hat) << endl;
}