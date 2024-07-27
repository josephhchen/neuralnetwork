#ifndef NEURON_H
#define NEURON_H

#include <vector>

class Neuron {
public:
    Neuron(int num_inputs);
    double activate(const std::vector<double>& inputs);
    void set_weights(const std::vector<double>& weights);
    void set_bias(double bias);

private:
    std::vector<double> weights;
    double bias;
    double sigmoid(double x);
    double dot_product(const std::vector<double>& v1, const std::vector<double>& v2);
};

#endif // NEURON_H
