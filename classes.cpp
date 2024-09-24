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

float getRandom(int range) {
    return ((float)rand() / RAND_MAX) * range * 2 - range;
}

Neuron::Neuron(int prevLayerNeurons)
{
    srand(time(NULL));
    this->prevLayerNeurons = prevLayerNeurons;
    float randRange = 10;
    this->value = 0;
    this->bias = decimalRounder(getRandom(randRange));
    this->weights = new float[prevLayerNeurons];

    for(int i = 0; i < prevLayerNeurons; i++){
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

        for (int j = 0; j < this->neurons[i]->prevLayerNeurons; j++) {
            cout << this->neurons[i]->weights[j] << ", ";
        }
        cout<<endl<<endl;
    }   
}

NeuralNet::NeuralNet(int layerCount, int* layerSizes){
    this->layerCount = layerCount;
    this->layerSizes = layerSizes;
    this->HidOutlayers = new Layer*[layerCount];
    for (int i = 0; i < layerCount; i++)
    {
        if (i == 0)
        {
            this->HidOutlayers[i] = new Layer(layerSizes[i], 0);
        }else{
            this->HidOutlayers[i] = new Layer(layerSizes[i], layerSizes[i-1]);
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

    cout << "Layer Count : " << this->layerCount << endl;
    printInColor("Layer Sizes: \n", "36"); // Cyan

    for (int i = 0; i < this->layerCount; i++) {
        cout << setw(4) << this->layerSizes[i]; // Formatting for better spacing
        if (i != this->layerCount - 1) {
            cout << " | ";
        }
        // Breaking line for better readability every 10 layers
        if ((i + 1) % 6 == 0) {
            cout << endl;
        }
    }
    cout<<endl<<endl;
}