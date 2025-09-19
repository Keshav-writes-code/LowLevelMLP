#ifndef CLASSES_H
#define CLASSES_H

namespace NeuralNet {
class Neuron {
public:
  float activation;
  float z;
  float bias;
  float *weights;
  int prevLayerNeurons_count;
  Neuron(int prevLayerNeurons_count);
  ~Neuron();
};

class Layer {
private:
  int size;

public:
  Neuron **neurons;
  float *activations;
  Layer(int size, int prevLayerSize);
  ~Layer();
  void forward_pass(float *inputs, int input_size);
  void showNeurons();
};

class MLP {
private:
  Layer **HidOutlayers;
  int hidOutLayerCount;
  int *hidOutLayerSizes;
  int inputLayerSize;
  int outputLayerSize;
  const float *predictions;

public:
  MLP(int inputLayerSize, int hidOutLayerCount, int *hidOutLayerSizes,
      int outputLayerSize);
  ~MLP();
  void describe();
  void resetNeuronsActivations();
  void feedForward(float *inputs, int inputSize);
  void predict(float **inputs, int inputSize, float **target, int targetSize,
               int samplesCount);
  float loss(float *target, int targetArr_size);
  float getParamTCostDerivative(float &param, float *inputArr, int inputSize,
                                float *targetArr, int targetArr_size);
  void backPropogate(float *inputArr, int inputSize, float *target,
                     int targetArr_size, float l_rate);
  void train(float **inputArr_2d, int input_elem_size, float **targetArr_2d,
             int target_elem_size, int items_count, int epochs, float l_rate);
  void printParamsCount();
  void constructLayer(int i);
};
} // namespace NeuralNet

#endif
