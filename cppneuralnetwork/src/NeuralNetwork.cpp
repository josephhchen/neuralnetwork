#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(const std::vector<int>& layers_config) {
    for (size_t i = 1; i < layers_config.size(); ++i) {
        layers.emplace_back(Layer(layers_config[i], layers_config[i - 1]));
    }
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& inputs) {
    std::vector<double> outputs = inputs;
    for (auto& layer : layers) {
        outputs = layer.feed_forward(outputs);
    }
    return outputs;
}
