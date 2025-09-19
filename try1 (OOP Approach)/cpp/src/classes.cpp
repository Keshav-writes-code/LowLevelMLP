#include "../include/NeuralNet/classes.h"
#include "../include/console/progressbar.h"
#include <chrono>  // For duration types
#include <cmath>   // for exp and round
#include <cstdlib> // for rand() and srand()
#include <iomanip> // for setw
#include <iostream>
#include <thread> // For threads
using namespace std;

float decimalRounder(float x) { return round(x * 100.0) / 100.0; }

float getRandom(float range) {
  return ((float)rand() / RAND_MAX) * range * 2 - range;
}

float sigmoid(float x) { return 1 / (1 + exp(-x)); }

float relu(float x) {
  if (x < 0)
    return 0;
  return x;
}
template <typename T> void printInColor(const T &val, const string &colorCode) {
  std::cout << "\033[" << colorCode << "m" << val << "\033[0m";
}
namespace NeuralNet {
Neuron::Neuron(int prevLayerNeurons_count) {
  this->prevLayerNeurons_count = prevLayerNeurons_count;
  float randRange = 1;
  this->activation = 0;
  this->z = 0;
  this->bias = decimalRounder(getRandom(randRange));
  this->weights = new float[prevLayerNeurons_count];

  for (int i = 0; i < prevLayerNeurons_count; i++) {
    this->weights[i] = decimalRounder(getRandom(randRange));
  }
}

Neuron::~Neuron() {
  delete[] this->weights;
  this->weights = nullptr;
  this->activation = 0;
  this->bias = 0;
}

Layer::Layer(int size, int prevLayerSize) {
  this->size = size;
  this->neurons = new Neuron *[size];

  for (int i = 0; i < size; i++) {
    this->neurons[i] = new Neuron(prevLayerSize);
  }
}

Layer::~Layer() {
  for (int i = 0; i < this->size; i++) {
    delete this->neurons[i];
  }
  delete[] this->neurons;
  this->neurons = nullptr;
}

void Layer::showNeurons() {
  for (int i = 0; i < this->size; i++) {
    std::cout << "Neuron - " << i << std::endl;
    std::cout << "value : " << this->neurons[i]->activation << std::endl;
    std::cout << "bias :- " << this->neurons[i]->bias << std::endl;
    std::cout << "weights: ";

    for (int j = 0; j < this->neurons[i]->prevLayerNeurons_count; j++) {
      std::cout << this->neurons[i]->weights[j] << ", ";
    }
    std::cout << endl << endl;
  }
}
void Layer::forward_pass(float *inputs, int input_size) {
  if (input_size != this->neurons[0]->prevLayerNeurons_count)
    throw runtime_error("Forward Pass: Expected Input Size was Not Received");

  for (int i = 0; i < this->size; i++) {
    Neuron *n = this->neurons[i];
    for (int j = 0; j < n->prevLayerNeurons_count; j++) {
      n->z += n->weights[j] * inputs[j];
    }
    n->z += n->bias;
    n->activation = sigmoid(n->z);
    this->activations[i] = n->activation;
    std::cout << "Helo : " << sigmoid(n->z) << "\n";
  }
}

void MLP::constructLayer(int i) {
  if (i == 0) {
    this->HidOutlayers[i] =
        new Layer(this->hidOutLayerSizes[i], this->inputLayerSize);
  } else if (i == hidOutLayerCount - 1) {
    this->HidOutlayers[i] =
        new Layer(this->outputLayerSize, this->hidOutLayerSizes[i - 1]);
    this->hidOutLayerSizes[i] = this->outputLayerSize;
  } else {
    this->HidOutlayers[i] =
        new Layer(this->hidOutLayerSizes[i], this->hidOutLayerSizes[i - 1]);
  }
}

MLP::MLP(int inputLayerSize, int hidOutLayerCount, int *hidOutLayerSizes,
         int outputLayerSize) {
  this->hidOutLayerCount = hidOutLayerCount;
  this->hidOutLayerSizes = hidOutLayerSizes;
  this->HidOutlayers = new Layer *[hidOutLayerCount];
  this->inputLayerSize = inputLayerSize;
  this->outputLayerSize = outputLayerSize;
  this->predictions = new float[outputLayerSize]();
  thread threads[hidOutLayerCount];
  for (int i = 0; i < hidOutLayerCount; i++) {
    threads[i] = thread(&MLP::constructLayer, this, i);
  }
  for (int i = 0; i < hidOutLayerCount; i++) {
    threads[i].join();
  }
}

void MLP::describe() {
  // Using ANSI escape codes to make the terminal output colorful
  printInColor("\n+-----------------------------------------+\n",
               "32"); // Green
  printInColor("|             Neural Network              |\n", "32");
  printInColor("+-----------------------------------------+\n", "32");

  std::cout << "Layer Count : " << this->hidOutLayerCount + 1 << endl;
  printInColor("Layer Sizes: \n", "36"); // Cyan

  std::cout << setw(4) << this->inputLayerSize << " | ";
  for (int i = 0; i < this->hidOutLayerCount; i++) {
    std::cout << setw(4)
              << this->hidOutLayerSizes[i]; // Formatting for better spacing
    if (i != this->hidOutLayerCount - 1) {
      std::cout << " | ";
    }
    // Breaking line for better readability every 10 layers
    if ((i + 1) % 6 == 0) {
      std::cout << endl;
    }
  }
  std::cout << endl << endl;
}

void MLP::resetNeuronsActivations() {
  // for Traversing Each Layer
  for (int i = 0; i < this->hidOutLayerCount; i++) {
    // for Traversing Each Neuron of a Layer
    for (int i2 = 0; i2 < this->hidOutLayerSizes[i]; i2++) {
      this->HidOutlayers[i]->neurons[i2]->activation = 0;
    }
  }
}

void MLP::feedForward(float *inputArr, int inputSize) {
  this->resetNeuronsActivations();

  // Tranver Layers

  float *intermediate_activations;
  for (int i = 0; i < this->hidOutLayerCount; i++) {
    Layer *c_layer = this->HidOutlayers[i];
    if (i == 0) {
      c_layer->forward_pass(inputArr, inputSize);
      intermediate_activations = c_layer->activations;
    } else {
      c_layer->forward_pass(intermediate_activations,
                            this->hidOutLayerSizes[i - 1]);
      intermediate_activations = c_layer->activations;
    }
  }

}

void MLP::predict(float **inputs, int inputSize, float **target, int targetSize,
                  int samplesCount) {
  float accuracy = 0;
  for (int j = 0; j < samplesCount; j++) {
    std::cout << "Inputs : ";
    for (int i = 0; i < inputSize; i++) {
      if (i != 0) {
        std::cout << ", ";
      }
      const float result = decimalRounder(inputs[j][i]);
      std::cout << result;
      if (i == inputSize - 1) {
        std::cout << endl;
      }
    }
    this->feedForward(inputs[j], inputSize);
    const int outputSize = this->hidOutLayerSizes[this->hidOutLayerCount - 1];
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
      const float result = decimalRounder(this->predictions[i]);
      string color = "32"; // Green
      if (i != maxIndex) {
        color = "31";
      } // Red
      else if (maxIndex == i && target[j][i] != 1) {
        color = "33";
      } // Yellow
      else if (maxIndex == i && target[j][i] == 1) {
        accuracy += 100 / samplesCount;
      }
      std::cout << "\033[" << color << "m" << "[" << to_string(i)
                << "] : " << result << " => " << target[j][i] << "\033[0m";
      std::cout << endl;
    }
  }
  std::cout << "Accuracy : " << accuracy << "%" << endl;
}

float MLP::loss(float *targetArr, int targetArr_size) {
  Layer *outLayer = this->HidOutlayers[this->hidOutLayerCount - 1];
  if (targetArr_size != this->outputLayerSize)
    throw runtime_error("Expected Target Array Size was Not Received");

  float loss = 0;
  for (int i = 0; i < this->outputLayerSize; i++) {
    cost += pow(outLayer->neurons[i]->activation - targetArr[i], 2);
  }
  return loss;
}
float MLP::getParamTCostDerivative(float &param, float *inputArr, int inputSize,
                                   float *targetArr, int targetArr_size) {
  float derivative = 0;
  this->feedForward(inputArr, inputSize);
  float diff = 0.0001;
  float previousCost = this->loss(targetArr, targetArr_size);
  param += diff;

  this->feedForward(inputArr, inputSize);
  float newCost = this->loss(targetArr, targetArr_size);

  this->feedForward(inputArr, inputSize);
  param -= diff;
  derivative = (newCost - previousCost) / diff;
  return derivative;
}

void MLP::backPropogate(float *inputArr, int inputSize, float *targetArr,
                        int targetArr_size, float l_rate) {
  this->feedForward(inputArr, inputSize);

  // For output layer Weights & Bias Adjustments
  const int last_hidden_layer_size =
      this->hidOutLayerSizes[this->hidOutLayerCount - 2];
  float *a_prev = new float[last_hidden_layer_size]();
  for (int i = 0; i < last_hidden_layer_size; i++) {
    a_prev[i] =
        this->HidOutlayers[this->hidOutLayerCount - 2]->neurons[i]->activation;
  }

  float *output_layer_deltas = new float[this->outputLayerSize]();
  for (int i = 0; i < this->outputLayerSize; i++) {
    Neuron *n = this->HidOutlayers[this->hidOutLayerCount - 1]->neurons[i];
    output_layer_deltas[i] = this->predictions[i] - targetArr[i];
    for (int j = 0; j < n->prevLayerNeurons_count; j++) {
      n->weights[j] -= l_rate * a_prev[j] * output_layer_deltas[i];
    }
    n->bias -= l_rate * output_layer_deltas[i];
  }

  // For hidden layer Weights Adjustments

  // for (int i = this->hidOutLayerCount - 1; i >= 0; i--) {
  //   // for Traversing Each Neuron of a Layer
  //   for (int i2 = 0; i2 < this->hidOutLayerSizes[i]; i2++) {
  //     float biasTCostDerivative = this->getParamTCostDerivative(
  //         this->HidOutlayers[i]->neurons[i2]->bias, inputArr, inputSize,
  //         targetArr, targetArr_size);
  //     this->HidOutlayers[i]->neurons[i2]->bias -=
  //         (biasTCostDerivative * this->lRate);
  //     // For traversing each Weight of current Neuron
  //     for (int i3 = 0;
  //          i3 < this->HidOutlayers[i]->neurons[i2]->prevLayerNeurons_count;
  //          i3++) {
  //       float weightTCostDerivative = 0;
  //       weightTCostDerivative = this->getParamTCostDerivative(
  //           this->HidOutlayers[i]->neurons[i2]->weights[i3], inputArr,
  //           inputSize, targetArr, targetArr_size);
  //       this->HidOutlayers[i]->neurons[i2]->weights[i3] -=
  //           (weightTCostDerivative * this->lRate);
  //     }
  //   }
  // }
}

void MLP::train(float **inputArr_2d, int input_elem_size, float **targetArr_2d,
                int target_elem_size, int items_count, int epochs,
                float l_rate) {
  std::cout << "\nTraining Progress :\n";
  for (int i = 0; i < epochs; i++) {
    Console::showProgressBar(epochs, i);
    std::cout.flush();
    for (int i2 = 0; i2 < items_count; i2++) {
      this->backPropogate(inputArr_2d[i2], input_elem_size, targetArr_2d[i2],
                          target_elem_size, l_rate);
    }
  }
}
void MLP::printParamsCount() {
  int weightsCount = this->inputLayerSize;
  int prevLayerNeuronsCount = this->inputLayerSize;
  int biasesCount = 0;

  for (int i = 0; i < this->hidOutLayerCount; i++) {
    weightsCount += this->hidOutLayerSizes[i] * prevLayerNeuronsCount;
    prevLayerNeuronsCount = this->hidOutLayerSizes[i];
    biasesCount += this->hidOutLayerSizes[i];
  }
  std::cout << "Weights Count : " << weightsCount << endl;
  std::cout << "Biases Count : " << biasesCount << endl;
}
} // namespace NeuralNet
