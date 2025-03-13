use color_print::{cformat, cprintln};
use rand::Rng;

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
    prev_layer_neurons_count: u16,
}
impl Layer {
    pub fn new(neurons_count: u16, prev_layer_neurons_count: u16) -> Layer {
        let mut neurons: Vec<Neuron> = Vec::new();
        for _ in 0..neurons_count {
            neurons.push(Neuron::new(prev_layer_neurons_count));
        }

        Layer {
            neurons,
            prev_layer_neurons_count,
        }
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
// TODO: Add remaining Structs like MLP,
pub struct Mlp {
    hid_out_layers: Vec<Layer>,
    input_layer_size: u16,
    lrate: f32,
}
impl Mlp {
    pub fn new(input_layer_size: u16, hid_out_layers_sizes: &Vec<u16>, lrate: f32) -> Mlp {
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
        println!();
    }
    pub fn reset_neurons_activations(&mut self) {
        for layer in self.hid_out_layers.iter_mut() {
            for neuron in layer.neurons.iter_mut() {
                neuron.value = 0.0;
            }
        }
    }
    pub fn feed_forward(&mut self, inputs: &Vec<f32>) -> Vec<f32> {
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
}
