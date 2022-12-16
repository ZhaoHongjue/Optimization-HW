#include <iostream>
#include <vector>
#include <fstream>
#include "Recorder.h"
using namespace std;

void Recorder::push_back(double train_loss, double val_loss, double grad_norm) {
    train_losses.push_back(train_loss);
    val_losses.push_back(val_loss);
    grad_norms.push_back(grad_norm);
}

void Recorder::to_csv() {
    ofstream outCSV;
    outCSV.open(this->ad);
    outCSV << ",train_loss,val_loss,grad_norm" << endl;
    for (int i = 0; i < train_losses.size(); i++) {
        outCSV << i << "," 
               << train_losses[i] << "," 
               << val_losses[i] << "," 
               << grad_norms[i] << endl;
    }
    outCSV.close();
}