#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Layer.h"
#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& layers_config);
    std::vector<double> predict(const std::vector<double>& inputs);

private:
    std::vector<Layer> layers;
};

#endif // NEURALNETWORK_H
