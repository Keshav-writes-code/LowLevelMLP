#include "../include/NeuralNet/classes.h"
#include "../include/console/progressbar.h"
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
float sigmoid_derivative(float val) { return val * (1 - val); }
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
  this->value = 0;
  this->bias = decimalRounder(getRandom(randRange));
  this->weights = new float[prevLayerNeurons_count];
  this->error = 0;

  for (int i = 0; i < prevLayerNeurons_count; i++) {
    this->weights[i] = decimalRounder(getRandom(randRange));
  }
}

Neuron::~Neuron() {
  delete[] this->weights;
  this->weights = nullptr;
  this->value = 0;
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
    std::cout << "value : " << this->neurons[i]->value << std::endl;
    std::cout << "bias :- " << this->neurons[i]->bias << std::endl;
    std::cout << "weights: ";

    for (int j = 0; j < this->neurons[i]->prevLayerNeurons_count; j++) {
      std::cout << this->neurons[i]->weights[j] << ", ";
    }
    std::cout << endl << endl;
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
         int outputLayerSize, float lRate) {
  this->hidOutLayerCount = hidOutLayerCount;
  this->hidOutLayerSizes = hidOutLayerSizes;
  this->HidOutlayers = new Layer *[hidOutLayerCount];
  this->inputLayerSize = inputLayerSize;
  this->outputLayerSize = outputLayerSize;
  this->lRate = lRate;
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
      this->HidOutlayers[i]->neurons[i2]->value = 0;
    }
  }
}

float *MLP::feedForward(float *inputArr, int inputSize) {
  this->resetNeuronsActivations();
  Layer *tempInputLayer = new Layer(inputLayerSize, 0);
  Layer *prevLayer = tempInputLayer;

  if (inputSize != this->inputLayerSize)
    throw runtime_error("Expected Input was Not Received");
  for (int i = 0; i < this->inputLayerSize; i++) {
    tempInputLayer->neurons[i]->value = inputArr[i];
  }

  // for Traversing Each Layer
  for (int i = 0; i < this->hidOutLayerCount; i++) {
    // for Traversing Each Neuron of a Layer
    for (int i2 = 0; i2 < this->hidOutLayerSizes[i]; i2++) {
      Neuron *cNeuron = this->HidOutlayers[i]->neurons[i2];
      float weightedSum = 0;
      // For traversing each Weight of current Neuron
      for (int i3 = 0; i3 < cNeuron->prevLayerNeurons_count; i3++) {
        weightedSum += prevLayer->neurons[i3]->value * cNeuron->weights[i3];
      }
      cNeuron->value += sigmoid(weightedSum) + cNeuron->bias;
      // TO DIplay Each Neuron's Final Activation in a Formatted way
      // std::cout<<"Neuron ["<<i<<"]"<<"["<<i2<<"] : "<<cNeuron->value<<endl;
    }
    prevLayer = this->HidOutlayers[i];
  }

  // for Returning output
  const int outputSize = this->hidOutLayerSizes[this->hidOutLayerCount - 1];
  float *out = new float[outputSize];
  for (int i = 0; i < outputSize; i++) {
    out[i] = prevLayer->neurons[i]->value;
  }
  delete tempInputLayer;
  return out;
}

void MLP::predict(float **inputs, int inputSize, float **target, int targetSize,
                  int samplesCount) {
  float accuracy = 0;
  for (int j = 0; j < samplesCount; j++) {
    std::cout << "\nInputs : ";
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
    float *outputs = this->feedForward(inputs[j], inputSize);
    const int outputSize = this->hidOutLayerSizes[this->hidOutLayerCount - 1];
    // Get highest Output
    float max = outputs[0];
    int maxIndex = 0;
    for (int i = 1; i < outputSize; i++) {
      if (outputs[i] > max) {
        max = outputs[i];
        maxIndex = i;
      }
    }
    std::cout << "Outputs : \n";
    for (int i = 0; i < outputSize; i++) {
      const float result = decimalRounder(outputs[i]);
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
    delete[] outputs;
  }
  std::cout << "Accuracy : " << accuracy << "%" << endl;
}

void MLP::backPropogate(float *inputArr, int inputSize, float *targetArr,
                        int targetArr_size) {

  const int out_layer_index = this->hidOutLayerCount - 1;
  // This layer represents the Layer ahead to current layer in the coming up
  // Algo
  // Initialy, it copies the data from output layer just to initalize it self
  Layer *permanent_pointer_to_temp_layer =
      new Layer(this->hidOutLayerSizes[out_layer_index],
                this->hidOutLayerSizes[out_layer_index - 1]);
  Layer *temp_pointer_to_next_layer = permanent_pointer_to_temp_layer;

  // NOTE: This is Just to Copy the output layer in the temp Layer and also
  // calculate error for each neuron.
  // Traversing Each Neurson of Output Layer only
  for (int i = 0; i < this->hidOutLayerSizes[out_layer_index]; i++) {
    Neuron *cNeuron = this->HidOutlayers[out_layer_index]->neurons[i];
    permanent_pointer_to_temp_layer->neurons[i]->error =
        (targetArr[i] - cNeuron->value) * sigmoid_derivative(cNeuron->value);

    // Traversing each weight of current neuron
    for (int i2 = 0; i2 < cNeuron->prevLayerNeurons_count; i2++) {
      permanent_pointer_to_temp_layer->neurons[i]->weights[i2] =
          cNeuron->weights[i2];
    }
  }

  // NOTE: Output of this Code Block: this is basically goning to Calculate
  // Error Values for each Neuron in the whole Network
  // Traversing Each Layer in reverse
  for (int i = this->hidOutLayerCount - 2; i >= 0; i--) {
    // Traverse to each neuron in current layer
    for (int i2 = 0; i2 < this->hidOutLayerSizes[i]; i2++) {
      float summed_error = 0;
      // Traverse to each neuron in the layer ahead to current layer
      // and calculating Error of Current layer neuron
      for (int i3 = 0; i3 < this->hidOutLayerSizes[i + 1]; i3++) {
        Neuron *nNeuron = temp_pointer_to_next_layer->neurons[i3];
        summed_error += nNeuron->weights[i2] * nNeuron->error;
      }
      this->HidOutlayers[i]->neurons[i2]->error =
          summed_error *
          sigmoid_derivative(this->HidOutlayers[i]->neurons[i2]->value);
    }
    temp_pointer_to_next_layer = this->HidOutlayers[i];
    if (i == this->hidOutLayerCount - 2) {
      delete permanent_pointer_to_temp_layer;
    }
  }

  Layer *tempInputLayer = new Layer(this->inputLayerSize, 0);
  Layer *prevLayer = tempInputLayer;
  for (int i = 0; i < this->inputLayerSize; i++) {
    tempInputLayer->neurons[i]->value = inputArr[i];
  }
  for (int i = 0; i < this->hidOutLayerCount; i++) {
    Layer *cLayer = this->HidOutlayers[i];
    for (int i2 = 0; i2 < this->hidOutLayerSizes[i]; i2++) {
      Neuron *cNeuron = cLayer->neurons[i2];
      for (int i3 = 0; i3 < cNeuron->prevLayerNeurons_count; i3++) {
        float *cWeight = &cNeuron->weights[i3];
        cNeuron->weights[i3] -=
            this->lRate * cNeuron->error * prevLayer->neurons[i3]->value;
      }
      cNeuron->bias -= this->lRate * cNeuron->error;
    }
    prevLayer = cLayer;
  }
  delete tempInputLayer;
}

void MLP::train(float **inputArr_2d, int input_elem_size, float **targetArr_2d,
                int target_elem_size, int items_count, int epochs) {
  std::cout << "\nTraining Progress :\n";
  for (int i = 0; i < epochs; i++) {
    Console::showProgressBar(epochs, i);
    std::cout.flush();
    for (int i2 = 0; i2 < items_count; i2++) {
      this->backPropogate(inputArr_2d[i2], input_elem_size, targetArr_2d[i2],
                          target_elem_size);
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
