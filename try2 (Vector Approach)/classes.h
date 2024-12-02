#ifndef CLASSES_H
#define CLASSES_H

class MLP
{
private:
    /* data */
public:
    MLP(int inputLayer_size, int hiddenLayers_count, int* hiddenLayers_sizes, int outputLayer_size);
    ~MLP();
    float*** weights;
    float** biases;
    float** values;
    void feedForward(int* inputs, int inputSize);
};



#endif