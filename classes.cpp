#include<iostream>
#include<cmath>
#include<cstdlib> // for rand() and srand()
#include<ctime>
#include<iomanip>
#include "classes.h"
using namespace std;

float decimalRounder(float x) {
    return round(x * 100.0) / 100.0;
}

float getRandom(float range) {
    return ((float)rand() / RAND_MAX) * range * 2 - range;
}

Neuron::Neuron(int prevLayerNeurons_count)
{
    this->prevLayerNeurons_count = prevLayerNeurons_count;
    float randRange = 1;
    this->value = 0;
    this->bias = decimalRounder(getRandom(randRange));
    this->weights = new float[prevLayerNeurons_count];

    for(int i = 0; i < prevLayerNeurons_count; i++){
        this->weights[i] = decimalRounder(getRandom(randRange));
    }
}

Neuron::~Neuron()
{
    delete[] this->weights;
    this->weights = nullptr;
    this->value = 0;
    this->bias = 0;
}

Layer::Layer(int size, int prevLayerSize)
{
    this->size = size;
    this->neurons = new Neuron*[size];

    for (int i = 0; i < size; i++)
    {
        this->neurons[i] = new Neuron(prevLayerSize);
    }
}


Layer::~Layer()
{
    for (int i = 0; i < this->size; i++) {
        delete this->neurons[i];
    }
    delete[] this->neurons;
    this->neurons = nullptr;
}

void Layer::showNeurons(){
    for (int i = 0; i < this->size; i++)
    {
        cout<<"Neuron - "<< i <<endl;
        cout<<"value : "<<this->neurons[i]->value<<endl;
        cout<<"bias :- "<<this->neurons[i]->bias<<endl;
        cout << "weights: ";

        for (int j = 0; j < this->neurons[i]->prevLayerNeurons_count; j++) {
            cout << this->neurons[i]->weights[j] << ", ";
        }
        cout<<endl<<endl;
    }   
}

NeuralNet::NeuralNet(int inputLayerSize, int hidOutLayerCount, int* hidOutLayerSizes, int outputLayerSize){
    this->hidOutLayerCount = hidOutLayerCount;
    this->hidOutLayerSizes = hidOutLayerSizes;
    this->HidOutlayers = new Layer*[hidOutLayerCount];
    this->inputLayerSize = inputLayerSize;
    this->outputLayerSize = outputLayerSize;
    for (int i = 0; i < hidOutLayerCount; i++)
    {
        if (i == 0)
        {
            this->HidOutlayers[i] = new Layer(hidOutLayerSizes[i], inputLayerSize);
        }
        else if(i == hidOutLayerCount-1){
            this->HidOutlayers[i] = new Layer(outputLayerSize, hidOutLayerSizes[i-1]);
            hidOutLayerSizes[i] = outputLayerSize;
        }
        else{
            this->HidOutlayers[i] = new Layer(hidOutLayerSizes[i], hidOutLayerSizes[i-1]);
        }
    }
}

void printInColor(const string& text, const string& colorCode) {
    cout << "\033[" << colorCode << "m" << text << "\033[0m";
}

void NeuralNet::describe() {
    // Using ANSI escape codes to make the terminal output colorful
    printInColor("\n+-----------------------------------------+\n", "32"); // Green
    printInColor("|             Neural Network              |\n", "32");
    printInColor("+-----------------------------------------+\n", "32");

    cout << "Layer Count : " << this->hidOutLayerCount << endl;
    printInColor("Layer Sizes: \n", "36"); // Cyan

    for (int i = 0; i < this->hidOutLayerCount; i++) {
        cout << setw(4) << this->hidOutLayerSizes[i]; // Formatting for better spacing
        if (i != this->hidOutLayerCount - 1) {
            cout << " | ";
        }
        // Breaking line for better readability every 10 layers
        if ((i + 1) % 6 == 0) {
            cout << endl;
        }
    }
    cout<<endl<<endl;
}

void NeuralNet::resetNeuronsActivations(){
    // for Traversing Each Layer
    for (int i = 0; i < this->hidOutLayerCount; i++)
    {
        // for Traversing Each Neuron of a Layer
        for (int i2 = 0; i2 < this->hidOutLayerSizes[i]; i2++)
        {
            this->HidOutlayers[i]->neurons[i2]->value = 0;
        }
    }
}

float* NeuralNet::feedForward(float* inputArr, int inputSize){
    this->resetNeuronsActivations();
    Layer* prevLayer = new Layer(inputLayerSize,0);
    if (inputSize != this->inputLayerSize) throw runtime_error("Expected Input was Not Received");
    for (int i = 0; i < this->inputLayerSize; i++){
        prevLayer->neurons[i]->value = inputArr[i];
    }

    // for Traversing Each Layer
    for (int i = 0; i < this->hidOutLayerCount; i++)
    {
        // for Traversing Each Neuron of a Layer
        for (int i2 = 0; i2 < this->hidOutLayerSizes[i]; i2++)
        {
            Neuron* cNeuron = this->HidOutlayers[i]->neurons[i2];
            // For traversing each Weight of current Neuron
            for (int i3 = 0; i3 < cNeuron->prevLayerNeurons_count; i3++)
            {
                cNeuron->value += prevLayer->neurons[i3]->value * cNeuron->weights[i3];
            }
            cNeuron->value += cNeuron->bias;
            // TO DIplay Each Neuron's Final Activation in a Formatted way
            // cout<<"Neuron ["<<i<<"]"<<"["<<i2<<"] : "<<cNeuron->value<<endl;
        }
        prevLayer = this->HidOutlayers[i];
    }
    
    // for Returning output
    const int outputSize = this->hidOutLayerSizes[this->hidOutLayerCount-1];
    float* out = new float[outputSize];
    for (int i = 0; i < outputSize; i++)
    {
        out[i] = prevLayer->neurons[i]->value;
    }
    return out;
}

void NeuralNet::predict(float* inputs, int inputSize){
    cout<<"Inputs : ";
    for (int i = 0; i < inputSize; i++)
    {
        if (i != 0){cout<<", ";}
        const float result = decimalRounder(inputs[i]);
        cout<<result;
        if (i == inputSize-1){cout<<endl;}
    }
    float* outputs = this->feedForward(inputs, inputSize);
    const int outputSize = this->hidOutLayerSizes[this->hidOutLayerCount-1];
    cout<<"Outputs : \n";
    for (int i = 0; i < outputSize; i++)
    {
        const float result = decimalRounder(outputs[i]);
        cout<<i<<" : "<<result<<endl;
    }
}

float NeuralNet::cost(float* targetArr, int targetArr_size){
    Layer* outLayer = this->HidOutlayers[this->hidOutLayerCount-1];
    if (targetArr_size != this->outputLayerSize) throw runtime_error("Expected Target Array Size was Not Received");

    float cost = 0;
    for (int i = 0; i < this->outputLayerSize; i++)
    {
        cost += pow( outLayer->neurons[i]->value - targetArr[i], 2);
    }
    return cost;
}

void NeuralNet::backPropogate(float* inputArr, int inputSize, float* targetArr, int targetArr_size){
    float cost = this->cost(targetArr, targetArr_size);
    

    this->HidOutlayers[0]->neurons[0]->weights[0] +=1;
    this->feedForward(inputArr, inputSize);
    cost = this->cost(targetArr, targetArr_size);
}