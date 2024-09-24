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
    Neuron** neurons;
    int size;
public:
    Layer(int size, int prevLayerSize);
    ~Layer();
    void showNeurons();
};

class NeuralNet{
    private:
    Layer** HidOutlayers;
    int layerCount;
    int* layerSizes;
    
    public:
    NeuralNet(int layerCount, int* layerSizes);
    ~NeuralNet();
    void describe();
};

#endif