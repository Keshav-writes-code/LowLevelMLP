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
    float *arr = new float[arrLen];
    for (int i = 0; i < arrLen; i++)
    {
        float randVal = ((float)rand() / RAND_MAX) * (max - min) + min;
        arr[i] = randVal;
    }
    return arr;
}
float** genRandom2DFloats(int rows, int cols, float min, float max) {
    // Allocate memory for the array of row pointers
    float** array2D = new float*[rows];
    // Generate each row using genRandomFloats
    for (int i = 0; i < rows; i++) {
        array2D[i] = genRandomFloats(cols, min, max);
    }
    return array2D;
}

float** getIdentityMatrix(int rows, int cols){
    float** matrix = new float*[rows];
    for (int i = 0; i < rows; i++) {
        matrix[i] = new float[cols];
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    return matrix;
}

void print2DArray(float** matrix, int rows, int cols){
    cout<<"2D Array : "<<endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

int main(){
    srand(time(NULL));
    const int inputLayerSize = 20;
    const int hidOutLayerCount = 3;
    const int outputLayerSize = 10;
    int * hidOutLayerSizes = genRandomInts(hidOutLayerCount, 3, 9);
    MLP* mlp1 = new MLP(inputLayerSize, hidOutLayerCount, hidOutLayerSizes, outputLayerSize, 0.3);


    // Dataset with labels
    const int samples_count = 10;
    float** inputs = genRandom2DFloats(samples_count, inputLayerSize, 0, 100);
    float** targets = getIdentityMatrix(samples_count, outputLayerSize);
}