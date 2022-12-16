#include <iostream>
#include <string>
#include <vector>
#include <eigen3/Eigen/Dense>

#include "Data.h"
#include "LinearModel.h"
#include "Optimizer.h"
#include "Recorder.h"
using namespace std;
using namespace Eigen;

void test_optimizer(int opt_mode) {
    double eps = 1e-10;
    double split_rate = 0.8;
    double lambda[6] = {0.0, 0.1, 0.5, 1.0, 5.0, 10.0};
    string ads[3] = {"abalone", "housing", "bodyfat"};
    Data data(split_rate);

    for (int i = 0; i < 3; i++) {
        data.read_data("./data/" + ads[i] + "_scale.txt");
        for (int j = 0; j < 6; j++) {
            if (opt_mode == 2) {
                for (int mode = 0; mode < 3; mode++) {
                    LinearModel lin(data);
                    quasiNetwon qN(lin, lambda[j]);
                    string ad = "./result/quasiNewton/" + ads[i] + "_" + to_string(mode) + "_"+ to_string_precision(lambda[j], 1) + ".csv";
                    Recorder r(ad);
                    qN.optimize(r, eps, mode);
                }
            }
            else if (opt_mode == 1) {
                for (int mode = 0; mode < 3; mode++) {
                    LinearModel lin(data);
                    ConjGrad CG(lin, lambda[j]);
                    string ad = "./result/ConjGrad/" + ads[i] + "_" + to_string(mode) + "_"+ to_string_precision(lambda[j], 1) + ".csv";
                    Recorder r(ad);
                    CG.optimize(r, eps, mode);
                }
            }
            else {
                LinearModel lin(data);
                GradDescent GD(lin, lambda[j]);
                string ad = "./result/Grad/" + ads[i] + "_"+ to_string_precision(lambda[j], 1) + ".csv";
                Recorder r(ad);
                GD.optimize(r, eps, 0);
            }
        }
    }
}

int main()
{   
    cout << "<=======================Start=======================>" << endl;
    
    int test_mode = 0;
    cin >> test_mode;
    test_optimizer(test_mode);
    
    
    cout << "<========================End========================>" << endl;
}