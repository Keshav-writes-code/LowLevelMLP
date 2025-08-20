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
    this.neurons.forEach((neuron) => neuron.calculate_activation(input));
  }
  get_last_layer_activations() {
    return this.neurons.map((n) => n.activation);
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
    input.forEach((x, i) => {
      this.z += x * this.weights_prev_layer[i];
    });
    this.z += this.bias;
  }
}

class out_layer {
  neurons: out_layer_neuron[];

  constructor(neurons_count: number, prev_layer_neurons_count: number) {
    this.neurons = Array.from(
      { length: neurons_count },
      () => new out_layer_neuron(prev_layer_neurons_count),
    );
  }

  calculate_activation(input: number[]) {
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
  }

  forward_propogation(input: number[]) {
    this.hidden_layers.forEach((layer) => layer.forward(input));

    let last_hidden_layer_activations =
      this.hidden_layers[
        this.hidden_layers.length - 1
      ].get_last_layer_activations();
    this.output_layer.calculate_activation(last_hidden_layer_activations);
    return this.output_layer.neurons.map((n) => n.activation);
  }
}
