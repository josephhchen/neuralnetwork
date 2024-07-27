#include "Neuron.h"
#include <cmath>
#include <cstdlib>
#include <ctime>

Neuron::Neuron(int num_inputs) {
    srand(time(0));
    weights.resize(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
        weights[i] = static_cast<double>(rand()) / RAND_MAX;
    }
    bias = static_cast<double>(rand()) / RAND_MAX;
}

double Neuron::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double Neuron::dot_product(const std::vector<double>& v1, const std::vector<double>& v2) {
    double result = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        result += v1[i] * v2[i];
    }
    return result;
}

double Neuron::activate(const std::vector<double>& inputs) {
    double z = dot_product(inputs, weights) + bias;
    return sigmoid(z);
}

void Neuron::set_weights(const std::vector<double>& new_weights) {
    weights = new_weights;
}

void Neuron::set_bias(double new_bias) {
    bias = new_bias;
}
