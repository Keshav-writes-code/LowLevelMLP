#ifndef CLASSES_H
#define CLASSES_H

enum class activations { relu, sigmoid, softmax };
namespace NeuralNet {
// Write Code here
class Layer {
private:
  int size;
  float *activation;
  float *z;
  float *bias;
  float **weights;
  int prev_layer_size;
  activations activation_function;
  void sigmoid();
  void relu();
  void softmax();
  friend class MLP;

public:
  Layer(int size, int prev_layer_size, activations activation_function);
  ~Layer();
  void forward_pass(const float *inputs);
  void show_neurons();
};

class MLP {
private:
  Layer **layers;
  int input_layer_size;
  int hidden_layers_count;
  int *hidden_layer_sizes;
  int output_layer_size;
  const float *predictions;

public:
  MLP(int input_layer_size, int hidden_layers_count, int *hidden_layer_sizes,
      int output_layer_size);
  ~MLP();
  void describe();
  void print_parameters_count();
  void feed_forward(float *inputs);
  void predict(float **feature_samples, float **target_samples,
               int samples_count);
};

} // namespace NeuralNet

#endif
