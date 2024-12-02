#include "classes.h"
#include<cstdlib> // for rand() and srand()
float getRandom(float range) {
  return ((float)rand() / RAND_MAX) * range * 2 - range;
}

MLP::MLP(int inputLayer_size, int hiddenLayers_count, int* hiddenLayers_sizes, int outputLayer_size, float lRate)
{
  // ------------------------------------------------------------------------
  // -------------------- Weight Initialization Part -----------------------
  // ------------------------------------------------------------------------
  this->weights = new float**[hiddenLayers_count +1]; // +1 for output neurons layer
  for (int i = 0; i < hiddenLayers_count; i++){
    this->weights[i] = new float*[hiddenLayers_sizes[i]];    
    for (int i2 = 0; i2 < hiddenLayers_sizes[i]; i2++){
      if (i==0){
        this->weights[i][i2] = new float[inputLayer_size];
        for (int i3 = 0; i3 < inputLayer_size; i3++){
          this->weights[i][i2][i3] = getRandom(1);
        }
      }else{
        this->weights[i][i2] = new float[hiddenLayers_sizes[i-1]];
        for (int i3 = 0; i3 < hiddenLayers_sizes[i-1]; i3++){
          this->weights[i][i2][i3] = getRandom(1);
        }
      }
    }
  }
  this->weights[hiddenLayers_count] = new float*[outputLayer_size];
  for (int i2 = 0; i2 < outputLayer_size; i2++){
    this->weights[hiddenLayers_count][i2] = new float[hiddenLayers_sizes[hiddenLayers_count-1]];
    for (int i3 = 0; i3 < hiddenLayers_sizes[hiddenLayers_count-1]; i3++){
      this->weights[hiddenLayers_count][i2][i3] = getRandom(1);
    }
  }
  // ------------------------------------------------------------------------
  // ------------------------------------------------------------------------

}

MLP::~MLP()
{
}
