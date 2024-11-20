#include<iostream>
#include "classes.h"

using namespace std;

int *genRandomInts(int arrLen, int min, int max)
{
    int *arr = (int *)malloc(arrLen * sizeof(int));
    for (int i = 0; i < arrLen; i++)
    {
        int randVal = rand() % (max - min + 1) + min;
        arr[i] = randVal;
    }
    return arr;
}
float *genRandomFloats(int arrLen, float min, float max)
{
    float *arr = (float *)malloc(arrLen * sizeof(float));
    for (int i = 0; i < arrLen; i++)
    {
        float randVal = ((float)rand() / RAND_MAX) * (max - min) + min;
        arr[i] = randVal;
    }
    return arr;
}
int main(){
    srand(time(NULL));
    const int inputLayerSize = 20;
    const int hidOutLayerCount = 20;
    const int outputLayerSize = 10;
    int * hidOutLayerSizes = genRandomInts(hidOutLayerCount, 3, 9);
    NeuralNet* NN = new NeuralNet(inputLayerSize, hidOutLayerCount, hidOutLayerSizes, outputLayerSize, 0.03);

    NN->describe();

    float* input1 = genRandomFloats(inputLayerSize, 0, 1000); 
    NN->predict(input1, inputLayerSize);

    float target1[] = {2,0,0,0,0,0,0,0,0,0};
    NN->backPropogate(input1, inputLayerSize, target1, 10, 100);

    NN->predict(input1, inputLayerSize);
}