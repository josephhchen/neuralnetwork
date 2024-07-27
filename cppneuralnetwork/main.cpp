#include "NeuralNetwork.h"
#include <iostream>

int main() {
    std::vector<int> layers_config = {2, 3, 1}; // 2 inputs, 3 hidden neurons, 1 output
    NeuralNetwork nn(layers_config);

    std::vector<double> inputs = {0.5, 0.8};
    std::vector<double> outputs = nn.predict(inputs);

    for (double output : outputs) {
        std::cout << "Output: " << output << std::endl;
    }

    return 0;
}
