#ifndef CLASSES_H
#define CLASSES_H


class Neuron
{
public:
    float value;
    float bias;
    float* weights;
    int prevLayerNeurons;
    Neuron(int prevLayerNeurons);
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
    public:
    NeuralNet(int hidOutLayerCount, int* hidOutLayerSizes);
    ~NeuralNet();
    void describe();
    void feedForward(float* inputs);
};

#endif