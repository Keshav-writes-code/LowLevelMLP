#include<iostream>
#include "classes.h"

using namespace std;

int *genRandomArr(int arrLen, int min, int max)
{
    int *arr = (int *)malloc(arrLen * sizeof(int));
    for (int i = 0; i < arrLen; i++)
    {
        int randVal = rand() % (max - min + 1) + min;
        arr[i] = randVal;
    }
    return arr;
}

int main(){
    srand(time(NULL));
    const int hidOutLayerCount = 20;
    const int inputLayerSize = 2;
    const int outputLayerSize = 2;
    int * hidOutLayerSizes = genRandomArr(hidOutLayerCount, 3, 9);
    NeuralNet* NN = new NeuralNet(inputLayerSize, hidOutLayerCount, hidOutLayerSizes, outputLayerSize);

    NN->describe();

    float hello[] = {0.4, 2.4}; 
    NN->predict(hello, 2);

    float target[] = {1,0};
    cout<<"\n"<<NN->cost(target, 2)<<"\n\n";
}