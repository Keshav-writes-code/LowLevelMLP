#include<iostream>
#include<cmath> // for exp and round
#include<cstdlib> // for rand() and srand()
#include<iomanip> // for setw
#include "classes.h"
#include <thread>
#include <chrono> // For duration types
using namespace std;

float decimalRounder(float x) {
    return round(x * 100.0) / 100.0;
}

float getRandom(float range) {
    return ((float)rand() / RAND_MAX) * range * 2 - range;
}

float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

float relu(float x) {
	if (x < 0) return 0;
	return x;
}
template<typename T>
void printInColor(const T& val, const string& colorCode) {
    cout << "\033[" << colorCode << "m" << val << "\033[0m";
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

void NeuralNet::constructLayer(int i){
    if (i == 0)
    {
        this->HidOutlayers[i] = new Layer(this->hidOutLayerSizes[i], this->inputLayerSize);
    }
    else if(i == hidOutLayerCount-1){
        this->HidOutlayers[i] = new Layer(this->outputLayerSize, this->hidOutLayerSizes[i-1]);
        this->hidOutLayerSizes[i] = this->outputLayerSize;
    }
    else{
        this->HidOutlayers[i] = new Layer(this->hidOutLayerSizes[i], this->hidOutLayerSizes[i-1]);
    }
}

NeuralNet::NeuralNet(int inputLayerSize, int hidOutLayerCount, int* hidOutLayerSizes, int outputLayerSize, float lRate){
    this->hidOutLayerCount = hidOutLayerCount;
    this->hidOutLayerSizes = hidOutLayerSizes;
    this->HidOutlayers = new Layer*[hidOutLayerCount];
    this->inputLayerSize = inputLayerSize;
    this->outputLayerSize = outputLayerSize;
		this->lRate = lRate;
    thread threads[hidOutLayerCount];
    for (int i = 0; i < hidOutLayerCount; i++)
    {
        threads[i] = thread(&NeuralNet::constructLayer, this, i); 
    }
    for (int i = 0; i < hidOutLayerCount; i++)
    {
        threads[i].join();
    }
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
						float weightedSum = 0;
            // For traversing each Weight of current Neuron
            for (int i3 = 0; i3 < cNeuron->prevLayerNeurons_count; i3++)
            {
                weightedSum += prevLayer->neurons[i3]->value * cNeuron->weights[i3];
            }
            cNeuron->value += sigmoid(weightedSum) + cNeuron->bias;
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

void NeuralNet::predict(float* inputs, int inputSize, float* target, int targetSize){
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
    // Get highest Output
    float max = outputs[0];
    int maxIndex = 0;
    for (int i = 1; i < outputSize; i++)
    {
        if (outputs[i] > max){
            max = outputs[i];
            maxIndex = i;
        }
    }

    cout<<"Outputs : \n";
    for (int i = 0; i < outputSize; i++)
    {
        const float result = decimalRounder(outputs[i]);
        string color = "32"; // Green
        if (i != maxIndex){color = "31";} // Red
        cout << "\033[" << color << "m" <<  "[" << to_string(i) << "] : " << result << " => " << target[i] << "\033[0m";
        cout<<endl;
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
float NeuralNet::getParamTCostDerivative(float& param, float* inputArr, int inputSize, float* targetArr, int targetArr_size){
    float derivative = 0;
    this->feedForward(inputArr, inputSize);
    float diff = 0.0001;
    float previousCost = this->cost(targetArr, targetArr_size);
    param += diff;

    this->feedForward(inputArr, inputSize);
    float newCost = this->cost(targetArr, targetArr_size);
    
    this->feedForward(inputArr, inputSize);
    param -= diff;
    derivative = (newCost - previousCost) / diff;
    return derivative;
}

void NeuralNet::backPropogate(float* inputArr, int inputSize, float* targetArr, int targetArr_size){
    // for Traversing Each Layer in reverse order
    for (int i = this->hidOutLayerCount - 1; i >= 0; i--)
    {
        // for Traversing Each Neuron of a Layer
        for (int i2 = 0; i2 < this->hidOutLayerSizes[i]; i2++)
        {
            float biasTCostDerivative = this->getParamTCostDerivative(this->HidOutlayers[i]->neurons[i2]->bias,inputArr, inputSize, targetArr, targetArr_size);
            this->HidOutlayers[i]->neurons[i2]->bias -= (biasTCostDerivative * this->lRate);
            // For traversing each Weight of current Neuron
            for (int i3 = 0; i3 < this->HidOutlayers[i]->neurons[i2]->prevLayerNeurons_count; i3++)
            {
                float weightTCostDerivative =0;
                weightTCostDerivative = this->getParamTCostDerivative(this->HidOutlayers[i]->neurons[i2]->weights[i3],inputArr, inputSize, targetArr, targetArr_size);
                this->HidOutlayers[i]->neurons[i2]->weights[i3] -= (weightTCostDerivative * this->lRate);           
            }
        }   
    }    
}

void NeuralNet::train(float** inputArr_2d, int input_elem_size, float** targetArr_2d, int target_elem_size, int items_count, int epochs){
    for (int i = 0; i < epochs; i++)
    {
        for (int i2 = 0; i2 < items_count; i2++)
        {
            this->backPropogate(inputArr_2d[i2], input_elem_size, targetArr_2d[i2], target_elem_size);
        }
    }
}
void NeuralNet::printParamsCount(){
    int weightsCount = this->inputLayerSize;
    int biasesCount= 0;

    for (int i = 1; i < this->hidOutLayerCount; i++)
    {
        weightsCount += this->hidOutLayerSizes[i]*this->hidOutLayerSizes[i-1];
    }
    for (int i = 0; i < this->hidOutLayerCount; i++)
    {
        biasesCount += this->hidOutLayerSizes[i];
    }
    cout<<"Weights Count : "<<weightsCount<<endl;
    cout<<"Biases Count : "<<biasesCount<<endl;
}