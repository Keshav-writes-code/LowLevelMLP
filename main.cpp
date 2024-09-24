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
    int layerCount = 100;
    int * layerSizes = genRandomArr(layerCount, 100, 1000);
    NeuralNet* NN = new NeuralNet(layerCount, layerSizes);

    NN->describe();
}