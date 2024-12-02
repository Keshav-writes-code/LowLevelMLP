#include "classes.h"

MLP::MLP(int inputLayer_size, int hiddenLayers_count, int* hiddenLayers_sizes, int outputLayer_size)
{
  // Initialize weight
  // also stores the output neurons weights
  this->weights = new float**[hiddenLayers_count +1]; // +1 for output neurons layer
  for (int i = 0; i < hiddenLayers_count; i++){
    this->weights[i] = new float*[hiddenLayers_sizes[i]];    
    for (int i2 = 0; i2 < hiddenLayers_sizes[i]; i2++){
      if (i==0){
        this->weights[i][i2] = new float[inputLayer_size];
      }else{
        this->weights[i][i2] = new float[hiddenLayers_sizes[i-1]];
      }
    }
  }
  this->weights[hiddenLayers_count] = new float*[outputLayer_size];
  for (int i2 = 0; i2 < outputLayer_size; i2++){
    this->weights[hiddenLayers_count][i2] = new float[hiddenLayers_sizes[hiddenLayers_count-1]];
  }
}

MLP::~MLP()
{
}
