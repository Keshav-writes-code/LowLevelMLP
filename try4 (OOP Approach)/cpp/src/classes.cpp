#include "../include/NeuralNet/classes.h"
#include "../include/console/progressbar.h"
#include <chrono>  // For duration types
#include <cmath>   // for exp and round
#include <cstdlib> // for rand() and srand()
#include <iomanip> // for setw
#include <iostream>
#include <thread> // For threads

float decimal_rounder(float x) { return round(x * 100.0) / 100.0; }

float get_random(float range) {
  return ((float)rand() / RAND_MAX) * range * 2 - range;
}

template <typename T>
void printInColor(const T &val, const std::string &colorCode) {
  std::cout << "\033[" << colorCode << "m" << val << "\033[0m";
}

namespace NeuralNet {

Layer::Layer(int size, int prev_layer_size, activations activation_function) {
  const float range = 0.01;

  this->size = size;
  this->prev_layer_size = prev_layer_size;
  this->activation_function = activation_function;

  this->activation = new float[size]();
  this->z = new float[size]();
  this->bias = new float[size];
  this->weights = new float *[size];
  for (int i = 0; i < size; i++) {
    this->bias[i] = get_random(range);
    this->weights[i] = new float[prev_layer_size];
    for (int j = 0; j < prev_layer_size; j++) {
      this->weights[i][j] = get_random(range);
    }
  }
}
Layer::~Layer() {
  delete[] this->activation;
  delete[] this->z;
  delete[] this->bias;
  for (int i = 0; i < this->size; i++) {
    delete[] this->weights[i];
  }
  delete[] this->weights;
}

void Layer::sigmoid() {
  for (int i = 0; i < this->size; i++) {
    this->activation[i] = 1 / (1 + exp(-this->z[i]));
  }
}
void Layer::relu() {
  for (int i = 0; i < this->size; i++) {
    this->activation[i] = this->z[i] > 0 ? this->z[i] : 0;
  }
}
void Layer::softmax() {
  float sum = 0;
  for (int i = 0; i < this->size; i++) {
    sum += exp(this->z[i]);
  }
  for (int i = 0; i < size; i++) {
    this->activation[i] = exp(this->z[i]) / sum;
  }
}

void Layer::forward_pass(const float *inputs) {
  for (int i = 0; i < this->size; i++) {
    for (int j = 0; j < this->prev_layer_size; j++) {
      this->z[i] += this->weights[i][j] * inputs[j];
    }
    this->z[i] += this->bias[i];

    switch (this->activation_function) {
    case activations::relu:
      this->relu();
      break;
    case activations::sigmoid:
      this->sigmoid();
      break;
    case activations::softmax:
      this->softmax();
      break;
    }
  }
}

MLP::MLP(int input_layer_size, int hidden_layers_count, int *hidden_layer_sizes,
         int output_layer_size) {
  this->input_layer_size = input_layer_size;
  this->hidden_layers_count = hidden_layers_count;
  this->hidden_layer_sizes = hidden_layer_sizes;
  this->output_layer_size = output_layer_size;
  this->predictions = nullptr;
  this->layers = new Layer *[hidden_layers_count + 1]; // +1 for output layer

  // Initialize Hidden layers
  for (int i = 0; i < hidden_layers_count; i++) {
    this->layers[i] =
        new Layer(hidden_layer_sizes[i],
                  (i - 1 < 0) ? input_layer_size : hidden_layer_sizes[i - 1],
                  activations::sigmoid);
  }
  // initialize output layer
  this->layers[this->hidden_layers_count] = new Layer(
      output_layer_size, hidden_layer_sizes[this->hidden_layers_count - 1],
      activations::softmax);
}
MLP::~MLP() {
  for (int i = 0; i < this->hidden_layers_count; i++) {
    delete[] this->layers[i];
  }
  delete[] this->layers;
}

void MLP::describe() {
  // Using ANSI escape codes to make the terminal output colorful
  printInColor("\n+-----------------------------------------+\n",
               "32"); // Green
  printInColor("|             Neural Network              |\n", "32");
  printInColor("+-----------------------------------------+\n", "32");

  std::cout << "Layer Count : " << this->hidden_layers_count + 1 << std::endl;
  printInColor("Layer Sizes: \n", "36"); // Cyan

  std::cout << std::setw(4) << this->input_layer_size << " | ";
  for (int i = 0; i < this->hidden_layers_count; i++) {
    std::cout << std::setw(4)
              << this->hidden_layer_sizes[i]; // Formatting for better spacing
    std::cout << " | ";
    // Breaking line for better readability every 10 layers
    if ((i + 1) % 6 == 0) {
      std::cout << std::endl;
    }
  }
  std::cout << std::setw(4)
            << this->output_layer_size; // Formatting for better spacing
  std::cout << " | ";
  // Breaking line for better readability every 10 layers
  std::cout << std::endl << std::endl;
}
void MLP::print_parameters_count() {
  int weightsCount = this->input_layer_size;
  int prevLayerNeuronsCount = this->input_layer_size;
  int biasesCount = 0;

  for (int i = 0; i < this->hidden_layers_count; i++) {
    weightsCount += this->hidden_layer_sizes[i] * prevLayerNeuronsCount;
    prevLayerNeuronsCount = this->hidden_layer_sizes[i];
    biasesCount += this->hidden_layer_sizes[i];
  }
  std::cout << "Weights Count : " << weightsCount << std::endl;
  std::cout << "Biases Count : " << biasesCount << std::endl;
}

void MLP::feed_forward(float *inputs) {
  this->layers[0]->forward_pass(inputs);
  const float *intermediate_activations = this->layers[0]->activation;
  for (int i = 1; i <= this->hidden_layers_count;
       i++) { // "<=" to account for the output layer
    Layer *c_layer = this->layers[i];
    c_layer->forward_pass(intermediate_activations);
    intermediate_activations = c_layer->activation;
  }
  this->predictions = intermediate_activations;
}

void MLP::predict(float **feature_samples, float **target_samples,
                  int samples_count) {
  float accuracy = 0;
  for (int j = 0; j < samples_count; j++) {
    std::cout << "Inputs : ";
    for (int i = 0; i < this->input_layer_size; i++) {
      if (i != 0) {
        std::cout << ", ";
      }
      const float result = decimal_rounder(feature_samples[j][i]);
      std::cout << result;
      if (i == this->input_layer_size - 1) {
        std::cout << std::endl;
      }
    }
    this->feed_forward(feature_samples[j]);
    const int outputSize = this->output_layer_size;
    // Get highest Output
    float max = this->predictions[0];
    int maxIndex = 0;
    for (int i = 1; i < outputSize; i++) {
      if (this->predictions[i] > max) {
        max = this->predictions[i];
        maxIndex = i;
      }
    }
    std::cout << "Outputs : \n";
    for (int i = 0; i < outputSize; i++) {
      const float result = decimal_rounder(this->predictions[i]);
      const char *color = "32"; // Green
      if (i != maxIndex) {
        color = "31";
      } // Red
      else if (maxIndex == i && target_samples[j][i] != 1) {
        color = "33";
      } // Yellow
      else if (maxIndex == i && target_samples[j][i] == 1) {
        accuracy += 100.0 / samples_count;
      }
      std::cout << "\033[" << color << "m" << "[" << i << "] : " << result
                << " => " << target_samples[j][i] << "\033[0m";
      std::cout << std::endl;
    }
  }
  std::cout << "Accuracy : " << accuracy << "%" << std::endl;
  ;
}

void MLP::back_propogate(float *features, float *targets, float l_rate) {}

} // namespace NeuralNet
