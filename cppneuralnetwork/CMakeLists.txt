cmake_minimum_required(VERSION 3.10)

# Set the project name
project(NeuralNetworkProject)

# Add include directories
include_directories(include)

# Add source files
set(SOURCES
    src/Neuron.cpp
    src/Layer.cpp
    src/NeuralNetwork.cpp
    main.cpp
)

# Add the executable
add_executable(NeuralNetworkProject ${SOURCES})
