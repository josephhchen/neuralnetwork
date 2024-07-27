#ifndef LAYER_H
#define LAYER_H

#include "Neuron.h"
#include <vector>

class Layer {
public:
    Layer(int num_neurons, int num_inputs_per_neuron);
    std::vector<double> feed_forward(const std::vector<double>& inputs);

private:
    std::vector<Neuron> neurons;
};

#endif // LAYER_H
