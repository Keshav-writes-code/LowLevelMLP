use color_print::{cprint, cprintln};
use rand::Rng;
mod console;
fn random_float(rand_range: u8) -> f32 {
    let mut rng = rand::rng();
    let random_val: f32 = rng.random::<f32>();

    (random_val * (rand_range as f32) * 2.0) - 1.0
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
struct Neuron {
    pub value: f32,
    pub bias: f32,
    pub weights: Vec<f32>,
}
impl Neuron {
    fn new(prev_layer_neurons_count: u16) -> Neuron {
        let rand_range: u8 = 1;
        let mut weights: Vec<f32> = Vec::new();
        for _ in 0..prev_layer_neurons_count {
            weights.push(random_float(rand_range));
        }

        Neuron {
            value: 0.0,
            bias: random_float(rand_range),
            weights,
        }
    }
}
pub struct Layer {
    neurons: Vec<Neuron>,
}
impl Layer {
    pub fn new(neurons_count: u16, prev_layer_neurons_count: u16) -> Layer {
        let mut neurons: Vec<Neuron> = Vec::new();
        for _ in 0..neurons_count {
            neurons.push(Neuron::new(prev_layer_neurons_count));
        }

        Layer { neurons }
    }
    pub fn show_neurons(&self) {
        for (i, neuron) in self.neurons.iter().enumerate() {
            println!("Neuron :- {}", i);
            println!("value :- {}", neuron.value);
            println!("bias :- {}", neuron.bias);
            println!("weights :");
            for weight in neuron.weights.iter() {
                print!("{:.2}, ", weight);
            }
            println!("\n");
        }
    }
}
pub struct Mlp {
    hid_out_layers: Vec<Layer>,
    input_layer_size: u16,
    lrate: f32,
}
impl Mlp {
    pub fn new(input_layer_size: u16, hid_out_layers_sizes: &[u16], lrate: f32) -> Mlp {
        let mut hid_out_layers: Vec<Layer> = Vec::new();
        let mut prev_layer_neurons_count: u16 = input_layer_size;
        for size in hid_out_layers_sizes.iter() {
            hid_out_layers.push(Layer::new(*size, prev_layer_neurons_count));
            prev_layer_neurons_count = *size;
        }
        Mlp {
            hid_out_layers,
            input_layer_size,
            lrate,
        }
    }
    pub fn describe(&self) {
        println!();
        cprintln!("<green>+------------------------------------+</>");
        cprintln!("<green>       Multi Layer Perceptron         </>");
        cprintln!("<green>+------------------------------------+</>");

        // +1 to also consider the input layer
        println!("Layer Count : {}", self.hid_out_layers.len() + 1);
        cprintln!("<cyan>Layer Sizes: </>");

        print!("{} | ", self.input_layer_size);
        for layer in self.hid_out_layers.iter() {
            print!("{} | ", layer.neurons.len());
        }
        println!("\n");
    }
    pub fn reset_neurons_activations(&mut self) {
        for layer in self.hid_out_layers.iter_mut() {
            for neuron in layer.neurons.iter_mut() {
                neuron.value = 0.0;
            }
        }
    }
    pub fn feed_forward(&mut self, inputs: &[f32]) -> Vec<f32> {
        self.reset_neurons_activations();
        if inputs.len() != self.input_layer_size as usize {
            panic!("Expected Input wasn't Received");
        }
        let mut prev_layer = &mut Layer::new(self.input_layer_size, 0);
        for (i, input) in inputs.iter().enumerate() {
            prev_layer.neurons[i].value = *input;
        }
        // For traversing each layer
        for layer in self.hid_out_layers.iter_mut() {
            // for traversing each neruons of the layer
            for neuron in layer.neurons.iter_mut() {
                let mut weighted_sum: f32 = 0.0;
                for (i, weight) in neuron.weights.iter().enumerate() {
                    weighted_sum += prev_layer.neurons[i].value * weight;
                }
                neuron.value += sigmoid(weighted_sum) + neuron.bias;
            }
            prev_layer = layer;
        }

        // For Returning Output
        let mut output: Vec<f32> = Vec::new();
        for neuron in prev_layer.neurons.iter() {
            output.push(neuron.value);
        }
        output
    }
    pub fn predict(&mut self, inputs: &[Vec<f32>], targets: &[Vec<f32>]) {
        let mut accuracy: f32 = 0.0;
        for (input_sample, target_sample) in inputs.iter().zip(targets.iter()) {
            println!("\nInputs : ");
            for input_sample_feature in input_sample.iter() {
                print!("{:.2}, ", input_sample_feature);
            }

            println!("\nOutputs : ");

            //Get the highest output
            let outputs = self.feed_forward(input_sample);
            let mut max: f32 = 0.0;
            let mut max_index: u8 = 0;
            for (i, output) in outputs.iter().enumerate() {
                if *output > max {
                    max = *output;
                    max_index = i as u8;
                }
            }

            // Show Results from feedfoward with color and target
            for (i, output) in outputs.iter().enumerate() {
                if i == max_index as usize && target_sample[i] == 1.0 {
                    cprintln!("<green>[{}] : {:.2} => {}</>", i, output, target_sample[i]);
                    accuracy += 100.0 / inputs.len() as f32;
                } else if i == max_index as usize && target_sample[i] != 1.0 {
                    cprintln!("<yellow>[{}] : {:.2} => {}</>", i, output, target_sample[i]);
                } else if i != max_index as usize {
                    cprint!("<red>[{}] : {:.2} => </>", i, output);
                    if target_sample[i] == 1.0 {
                        cprintln!("<green>{}</>", target_sample[i]);
                    } else {
                        cprintln!("<red>{}</>", target_sample[i]);
                    }
                }
            }
        }
        println!("Accuracy : {:.2}%", accuracy);
    }
    pub fn cost(&self, targets: &[f32]) -> f32 {
        let mut cost: f32 = 0.0;
        let out_layer = self
            .hid_out_layers
            .last()
            .expect("Hid Out Layers Should not be Empty");
        for (neuron, target) in out_layer.neurons.iter().zip(targets.iter()) {
            cost += (neuron.value - target).powf(2.0);
        }
        cost
    }
    pub fn get_param_to_cost_derivative(
        &mut self,
        layer_id: usize,
        neuron_id: usize,
        weight_id: Option<usize>,
        is_bias: bool,
        inputs: &[f32],
        targets: &[f32],
    ) -> f32 {
        let diff: f32 = 0.0001;

        // calculate cost before any nudge
        self.feed_forward(inputs);
        let prev_cost = self.cost(targets);

        // calculate cost after the nudge
        if is_bias {
            self.hid_out_layers[layer_id].neurons[neuron_id].bias += diff;
        } else {
            self.hid_out_layers[layer_id].neurons[neuron_id].weights
                [weight_id.expect("Weight Id Expected but not Provided")] += diff;
        }
        self.feed_forward(inputs);
        let new_cost = self.cost(targets);

        // reset the nudge to the bias and neurons values
        if is_bias {
            self.hid_out_layers[layer_id].neurons[neuron_id].bias -= diff;
        } else {
            self.hid_out_layers[layer_id].neurons[neuron_id].weights
                [weight_id.expect("Weight Id Expected but not Provided")] -= diff;
        }

        //return output
        (new_cost - prev_cost) / diff
    }

    pub fn back_propogate(&mut self, inputs: &[f32], targets: &[f32]) {
        // go through each layer in reverse order
        for layer_id in 0..self.hid_out_layers.len() {
            // go through each neuron in the layer
            for neuron_id in 0..self.hid_out_layers[layer_id].neurons.len() {
                let bias_to_cost_derivative = self
                    .get_param_to_cost_derivative(layer_id, neuron_id, None, true, inputs, targets);
                self.hid_out_layers[layer_id].neurons[neuron_id].bias -=
                    self.lrate * bias_to_cost_derivative;
                // going through each weight of the neuron
                for weight_id in 0..self.hid_out_layers[layer_id].neurons[neuron_id]
                    .weights
                    .len()
                {
                    let weight_to_cost_derivative: f32 = self.get_param_to_cost_derivative(
                        layer_id,
                        neuron_id,
                        Some(weight_id),
                        false,
                        inputs,
                        targets,
                    );
                    self.hid_out_layers[layer_id].neurons[neuron_id].weights[weight_id] -=
                        self.lrate * weight_to_cost_derivative;
                }
            }
        }
    }
    pub fn train(&mut self, inputs: &[Vec<f32>], targets: &[Vec<f32>], epochs: usize) {
        println!("Traning Progress : ");
        for i in 0..epochs {
            console::update_progress_bar(i, epochs);
            for (input, target) in inputs.iter().zip(targets.iter()) {
                self.back_propogate(input, target);
            }
        }
    }
    pub fn print_params_count(&mut self) {
        let mut weights_count: u16 = 0;
        let mut biases_count: u16 = 0;
        let mut prev_layer_neuron_count = self.input_layer_size;
        for layer in self.hid_out_layers.iter() {
            weights_count += prev_layer_neuron_count * layer.neurons.len() as u16;
            biases_count += layer.neurons.len() as u16;
            prev_layer_neuron_count = layer.neurons.len() as u16;
        }
        println!("Weights Count : {}", weights_count);
        println!("Biases Count : {}", biases_count);
    }
}
