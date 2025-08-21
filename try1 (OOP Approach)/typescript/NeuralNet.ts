import { sigmoid } from "./utils";

class hidden_layer_neuron {
  activation: number;
  weights_prev_layer: number[];
  bias: number;
  constructor(prev_layer_neurons: number) {
    this.activation = 0;
    this.weights_prev_layer = Array.from({ length: prev_layer_neurons }, () =>
      Math.random(),
    );
    this.bias = 0;
  }
  calculate_activation(inputs: number[]) {
    let z = 0;
    inputs.forEach((x, i) => {
      z += x * this.weights_prev_layer[i];
    });
    z += this.bias;
    this.activation = sigmoid(z);
    return this.activation;
  }
}

class hidden_layer {
  neurons: hidden_layer_neuron[];
  constructor(neurons_count: number, prev_layer_neurons: number) {
    this.neurons = Array.from(
      { length: neurons_count },
      () => new hidden_layer_neuron(prev_layer_neurons),
    );
  }

  forward(input: number[]) {
    return this.neurons.map((neuron) => neuron.calculate_activation(input));
  }
}

class out_layer_neuron {
  activation: number;
  z: number;
  weights_prev_layer: number[];
  bias: number;

  constructor(prev_layer_neurons_count: number) {
    this.weights_prev_layer = Array.from(
      { length: prev_layer_neurons_count },
      () => Math.random(),
    );
    this.activation = 0;
    this.z = 0;
    this.bias = 0;
  }

  calculate_z(input: number[]) {
    this.z = this.bias;
    input.forEach((x, i) => {
      this.z += x * this.weights_prev_layer[i];
    });
  }
}

class out_layer {
  neurons: out_layer_neuron[];
  input: number[];

  constructor(neurons_count: number, prev_layer_neurons_count: number) {
    this.neurons = Array.from(
      { length: neurons_count },
      () => new out_layer_neuron(prev_layer_neurons_count),
    );
  }

  calculate_activation(input: number[]) {
    this.input = input;
    let sum_of_z_exp = 0;
    this.neurons.forEach((neuron) => {
      neuron.calculate_z(input);
      sum_of_z_exp += Math.exp(neuron.z);
    });

    this.neurons.forEach((neuron) => {
      neuron.activation = Math.exp(neuron.z) / sum_of_z_exp;
    });
  }
}

export class MLP {
  hidden_layers: hidden_layer[];
  output_layer: out_layer;
  predictions: number[];
  constructor(
    input_layer_size: number,
    hidden_layer_sizes: number[],
    out_layer_size: number,
  ) {
    this.hidden_layers = hidden_layer_sizes.map(
      (size, i) =>
        new hidden_layer(size, hidden_layer_sizes[i - 1] || input_layer_size),
    );
    this.output_layer = new out_layer(
      out_layer_size,
      hidden_layer_sizes[hidden_layer_sizes.length - 1],
    );
    this.predictions = Array.from({ length: out_layer_size }, () => 0);
  }

  get_last_hidden_layer_activations() {
    return this.hidden_layers[this.hidden_layers.length - 1].neurons.map(
      (n) => n.activation,
    );
  }
  forward_propogation(input: number[]) {
    this.hidden_layers.forEach((layer, i) => {
      let intermedia_activations: number[] = [];

      // Pass Inputs to the 1st Layer
      if (i == 0) {
        intermedia_activations = layer.forward(input);
      } else {
        intermedia_activations = layer.forward(intermedia_activations);
      }
    });

    this.output_layer.calculate_activation(
      this.get_last_hidden_layer_activations(),
    );

    this.predictions = this.output_layer.neurons.map((n) => n.activation);
    return this.predictions;
  }

  loss(target: number[]) {
    // NOTE: Assumes you have already done the Forward Propogation
    // NOTE: Assimes the Target is in the One Hot Encoding Format

    let loss = 0;
    target.forEach((y, i) => {
      loss -= y * Math.log(this.output_layer.neurons[i].activation);
    });
    return loss;
  }

  // Single Sample single adjustment
  backpropogate(input: number[], target: number[], l_rate: number) {
    this.forward_propogation(input);

    // For output layer Weights Adjustments
    for (const [i, x] of this.get_last_hidden_layer_activations().entries()) {
      this.output_layer.neurons[i].weights_prev_layer =
        this.output_layer.neurons[i].weights_prev_layer.map((w, j) => {
          let gradient = x * (this.predictions[j] - target[j]);
          return w - l_rate * gradient;
        });
    }
  }
}
