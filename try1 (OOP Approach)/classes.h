#ifndef CLASSES_H
#define CLASSES_H


class Neuron
{
public:
    float value;
    float bias;
    float* weights;
    int prevLayerNeurons_count;
    Neuron(int prevLayerNeurons_count);
    ~Neuron();
};

class Layer
{
private:
    int size;
public:
    Neuron** neurons;
    Layer(int size, int prevLayerSize);
    ~Layer();
    void showNeurons();
};

class NeuralNet{
    private:
    Layer** HidOutlayers;
    int hidOutLayerCount;
    int* hidOutLayerSizes;
    int inputLayerSize;
    int outputLayerSize;
    float lRate;
    public:
    NeuralNet(int inputLayerSize, int hidOutLayerCount, int* hidOutLayerSizes, int outputLayerSize, float lRate);
    ~NeuralNet();
    void describe();
    void resetNeuronsActivations();
    float* feedForward(float* inputs, int inputSize);
    void predict(float** inputs, int inputSize, float** target, int targetSize, int samplesCount);
    float cost(float* target, int targetArr_size);
    float getParamTCostDerivative(float& param, float* inputArr, int inputSize, float* targetArr, int targetArr_size);
    void backPropogate(float* inputArr, int inputSize, float* target, int targetArr_size);
    void train(float** inputArr_2d, int input_elem_size, float** targetArr_2d, int target_elem_size, int items_count, int epochs);
    void printParamsCount();
    void constructLayer(int i);
};

#endif