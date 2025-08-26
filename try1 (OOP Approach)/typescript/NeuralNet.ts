import { sigmoid } from "./utils";

class hidden_layer_neuron {
  activation: number;
  weights_prev_layer: number[];
  bias: number;
  z: number;
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
    this.z = z;
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

    // For output layer Weights & Bias Adjustments
    let a_prev = this.get_last_hidden_layer_activations();
    let output_layer_deltas: number[] = [];
    this.output_layer.neurons.forEach((neuron, i) => {
      output_layer_deltas[i] = this.predictions[i] - target[i];
      neuron.weights_prev_layer = neuron.weights_prev_layer.map(
        (w, j) => w - l_rate * a_prev[j] * output_layer_deltas[i],
      );
      neuron.bias -= l_rate * output_layer_deltas[i];
    });

    // for hidden layer Weights Adjustment
    let next_layer_deltas = output_layer_deltas;
    for (
      let layer_idx = this.hidden_layers.length - 1;
      layer_idx >= 0;
      layer_idx--
    ) {
      const current_layer = this.hidden_layers[layer_idx];
      const next_layer =
        layer_idx == this.hidden_layers.length - 1
          ? this.output_layer
          : this.hidden_layers[layer_idx + 1];
      const a_prev =
        layer_idx == 0
          ? input
          : this.hidden_layers[layer_idx - 1].neurons.map((n) => n.activation);
      const current_layer_deltas: number[] = [];
      current_layer.neurons.forEach((n, i) => {
        let error_sum = 0;
        next_layer.neurons.forEach((n_next, j) => {
          error_sum += next_layer_deltas[j] * n_next.weights_prev_layer[i];
        });

        const activation_derivative = n.z > 0 ? 1 : 0;
        const delta = activation_derivative * error_sum;
        current_layer_deltas.push(delta);

        n.weights_prev_layer = n.weights_prev_layer.map(
          (w, j) => w - l_rate * a_prev[j] * delta,
        );
        n.bias -= l_rate * delta;
      });
      next_layer_deltas = current_layer_deltas;
    }
  }
}
